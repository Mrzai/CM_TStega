from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import opts
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
from models.AoAModel import AoAModel
from models.SecretModel import SecretEncoder, SecretExtractor
from optimize_utils import ModelOptimizer
from misc.utils import (
    get_model_size, 
    benchmark_inference_speed,
    OptimizationLogger,
    save_optimized_model
)

def main():
    # 复用eval.py的参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='',
                    help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
    parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
    parser.add_argument('--secret_size', type=int, default=10,
                    help='the number of secret bit')
    
    # 添加优化相关参数
    parser.add_argument('--prune_amount', type=float, default=0.3,
                    help='Amount of parameters to prune')
    parser.add_argument('--distill_epochs', type=int, default=5,
                    help='Number of epochs for knowledge distillation')
    
    # 添加eval.py中的其他参数
    opts.add_eval_options(parser)
    opt = parser.parse_args()

    # 加载infos
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # 复用eval.py的参数处理逻辑
    replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
    ignore = ['start_from']

    for k in vars(infos['opt']).keys():
        if k in replace:
            setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
        elif k not in ignore:
            if not k in vars(opt):
                vars(opt).update({k: vars(infos['opt'])[k]})

    vocab = infos['vocab']
    opt.vocab = vocab 

    # 初始化模型
    cap_model = AoAModel(opt).cuda()
    sec_encoder = SecretEncoder(opt.secret_size).cuda()
    sec_extractor = SecretExtractor(opt).cuda()
    del opt.vocab

    # 加载模型权重
    cap_model.load_state_dict(torch.load(os.path.join(opt.model_path, 'cap_model-best.pth')))
    sec_encoder.load_state_dict(torch.load(os.path.join(opt.model_path, 'sec_encoder-best.pth')))
    sec_extractor.load_state_dict(torch.load(os.path.join(opt.model_path, 'sec_extractor-best.pth')))

    # 初始化数据加载器
    loader = DataLoader(opt)
    loader.ix_to_word = infos['vocab']

    # 初始化评估标准
    crit = utils.LanguageModelCriterion()
    
    # 初始化日志记录器
    logger = OptimizationLogger(os.path.join(opt.model_path, 'optimization_logs'))

    # 1. 评估原始模型
    print("\nEvaluating original model...")
    loss, split_predictions, lang_stats = eval_utils.eval_split(cap_model, sec_encoder, 
                                                              sec_extractor, crit, loader, 
                                                              vars(opt))
    logger.log_metrics('original', {
        'size': get_model_size(cap_model),
        'language_metrics': lang_stats
    })
    print('Original model metrics:', lang_stats)

    # 初始化模型优化器
    optimizer = ModelOptimizer(cap_model, loader)

    # 2. 量化优化
    print("\nQuantizing model...")
    cap_model_quantized = optimizer.quantize()
    loss, split_predictions, lang_stats = eval_utils.eval_split(cap_model_quantized,
                                                              sec_encoder, sec_extractor,
                                                              crit, loader, vars(opt))
    logger.log_metrics('quantized', {
        'size': get_model_size(cap_model_quantized),
        'language_metrics': lang_stats
    })
    print('Quantized model metrics:', lang_stats)

    # 3. 剪枝优化
    print("\nPruning model...")
    cap_model_pruned = optimizer.prune(amount=opt.prune_amount)
    loss, split_predictions, lang_stats = eval_utils.eval_split(cap_model_pruned,
                                                              sec_encoder, sec_extractor,
                                                              crit, loader, vars(opt))
    logger.log_metrics('pruned', {
        'size': get_model_size(cap_model_pruned),
        'language_metrics': lang_stats
    })
    print('Pruned model metrics:', lang_stats)

    # 4. 知识蒸馏
    print("\nDistilling model...")
    cap_model_distilled = optimizer.distill(loader, epochs=opt.distill_epochs)
    loss, split_predictions, lang_stats = eval_utils.eval_split(cap_model_distilled,
                                                              sec_encoder, sec_extractor,
                                                              crit, loader, vars(opt))
    logger.log_metrics('distilled', {
        'size': get_model_size(cap_model_distilled),
        'language_metrics': lang_stats
    })
    print('Distilled model metrics:', lang_stats)

    # 保存优化报告
    logger.print_summary()
    logger.save_report()

    # 保存优化后的模型
    save_optimized_model(
        cap_model_distilled,
        os.path.join(opt.model_path, 'cap_model_optimized.pth')
    )

if __name__ == "__main__":
    main()