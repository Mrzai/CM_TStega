import torch
import torch.nn as nn
from optimize_utils import ModelOptimizer
from models.AoAModel import AoAModel
import opts
import misc.utils as utils
import os

class DummyDataLoader:
    """模拟数据加载器"""
    def __init__(self, batch_size=16, seq_length=16, feature_size=2048):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.feature_size = feature_size
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # 生成模拟数据
        fc_feats = torch.randn(self.batch_size, self.feature_size)
        att_feats = torch.randn(self.batch_size, 36, self.feature_size)
        att_masks = torch.ones(self.batch_size, 36)
        labels = torch.randint(0, 9000, (self.batch_size, self.seq_length))
        return fc_feats, att_feats, labels, att_masks

def test_optimization():
    # 1. 设置基本参数
    parser = opts.parse_opt()
    opt = parser.parse_args([])  # 空参数列表
    opt.vocab_size = 9000  # 模拟词汇表大小
    opt.seq_length = 16
    opt.fc_feat_size = 2048
    opt.att_feat_size = 2048
    opt.att_hid_size = 512
    
    # 2. 创建模型
    print("Creating model...")
    model = AoAModel(opt)
    
    # 3. 创建模拟数据加载器
    dummy_loader = DummyDataLoader()
    
    # 4. 创建优化器
    print("Initializing optimizer...")
    optimizer = ModelOptimizer(model, dummy_loader)
    
    # 5. 测试量化
    print("\nTesting quantization...")
    try:
        quantized_model = optimizer.quantize()
        print("Quantization successful!")
        print(f"Original model size: {utils.get_model_size(model)}MB")
        print(f"Quantized model size: {utils.get_model_size(quantized_model)}MB")
    except Exception as e:
        print("Quantization failed:", str(e))
    
    # 6. 测试剪枝
    print("\nTesting pruning...")
    try:
        pruned_model = optimizer.prune(amount=0.3)
        print("Pruning successful!")
        print(f"Pruned model size: {utils.get_model_size(pruned_model)}MB")
    except Exception as e:
        print("Pruning failed:", str(e))
    
    # 7. 测试知识蒸馏
    print("\nTesting distillation...")
    try:
        distilled_model = optimizer.distill(dummy_loader, epochs=2)
        print("Distillation successful!")
    except Exception as e:
        print("Distillation failed:", str(e))
    
    # 8. 测试推理
    print("\nTesting inference...")
    try:
        fc_feats, att_feats, _, att_masks = next(iter(dummy_loader))
        with torch.no_grad():
            # 测试原始模型
            output_original = model(fc_feats, att_feats, att_masks)
            print("Original model inference successful!")
            
            # 测试量化模型
            if 'quantized_model' in locals():
                output_quantized = quantized_model(fc_feats, att_feats, att_masks)
                print("Quantized model inference successful!")
            
            # 测试剪枝模型
            if 'pruned_model' in locals():
                output_pruned = pruned_model(fc_feats, att_feats, att_masks)
                print("Pruned model inference successful!")
            
            # 测试蒸馏模型
            if 'distilled_model' in locals():
                output_distilled = distilled_model(fc_feats, att_feats, att_masks)
                print("Distilled model inference successful!")
    except Exception as e:
        print("Inference testing failed:", str(e))

if __name__ == "__main__":
    test_optimization() 