from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import argparse
import models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloader import *
import misc.utils as utils
from models.AoAModel import AoAModel
from optimize_utils import (
    SimpleStudentModel, 
    move_to_device,
    find_latest_checkpoint,
    save_checkpoint,
    load_checkpoint
)
import os
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import datetime
import time
from torch.nn.utils import prune
from models.SecretModel import SecretEncoder, SecretExtractor

# 重新实现 FeatureLoader 以避免使用 lambda
class FeatureLoader:
    """特征加载器"""
    def __init__(self, input_dir, input_type):
        self.input_dir = input_dir
        self.input_type = input_type
        self.handlers = {}
        
    def get(self, key):
        # 如果已经加载过，直接返回
        if key in self.handlers:
            return self.handlers[key]
        
        # 否则加载特征
        feat_path = os.path.join(self.input_dir, str(key))
        if self.input_type == 'fc':
            feat = np.load(feat_path + '.npy')
        elif self.input_type == 'att':
            feat = np.load(feat_path + '.npz')['feat']
        else:
            raise ValueError(f'Unknown input type: {self.input_type}')
            
        # 缓存并返回
        self.handlers[key] = feat
        return feat
    
    def __del__(self):
        # 清理缓存
        self.handlers.clear()

# 将 CaptionDataset 移到全局作用域
class CaptionDataset(Dataset):
    def __init__(self, opt, infos, split='train'):
        self.opt = opt
        self.split = split
        self.label_path = opt.input_label_h5
        
        # 从infos中获取序列长度
        self.seq_length = infos['opt'].seq_length
        
        # 使用新的特征加载器
        self.fc_loader = FeatureLoader(self.opt.input_fc_dir, 'fc')
        self.att_loader = FeatureLoader(self.opt.input_att_dir, 'att')
        
        # 获取数据索引
        self.info = json.load(open(self.opt.input_json))
        self.iterator = 0
        self.split_ix = {}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == split:
                if split not in self.split_ix:
                    self.split_ix[split] = []
                self.split_ix[split].append(ix)
        
        self.length = len(self.split_ix[split])
        self.vocab_size = len(infos['vocab'])
        print('Dataset length for split %s: %d' % (split, self.length))
        
        # 不在初始化时打开h5py文件
        self._label_file = None
        
        # 检查h5py文件结构
        with h5py.File(self.label_path, 'r') as f:
            print(f"\nH5 file structure for {split}:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
            f.visititems(print_structure)
            
            # 确定正确的键名
            if 'labels' in f:
                self.label_key = 'labels'
            else:
                available_keys = list(f.keys())
                print(f"\nAvailable keys in h5 file: {available_keys}")
                # 尝试找到包含label的键
                label_keys = [k for k in available_keys if 'label' in k.lower()]
                
                if label_keys:
                    self.label_key = label_keys[0]
                else:
                    raise ValueError(f"Cannot find label data in h5 file. Available keys: {available_keys}")
    
    @property
    def label(self):
        # 懒加载h5py文件
        if self._label_file is None:
            self._label_file = h5py.File(self.label_path, 'r')
        return self._label_file
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        try:
            # 增加超时时间
            start_time = time.time()
            timeout = 60  # 增加到60秒
            
            ix = self.split_ix[self.split][idx]
            
            # 加载特征并验证
            img_id = str(self.info['images'][ix]['id'])
            
            # 加载特征
            fc_feat = self.fc_loader.get(img_id)
            att_feat = self.att_loader.get(img_id)
            
            if fc_feat is None or att_feat is None:
                print(f"Warning: Failed to load features for image {img_id}")
                # 返回一个默认的空特征而不是None
                return self._get_default_item()
            
            # 加载标签
            label = np.array(self.label[self.label_key][ix])
            if label is None:
                print(f"Warning: Label is None for index {ix}")
                return self._get_default_item()
            
            label_length = int(self.label['label_length'][ix])
            if not (0 < label_length <= len(label)):
                print(f"Warning: Invalid label_length {label_length} for index {ix}")
                return self._get_default_item()
            
            # 类型转换
            label = label.astype(np.int64)
            mask = np.zeros_like(label, dtype=np.float32)
            mask[:label_length] = 1
            
            # 转换为tensor
            fc_feat = torch.from_numpy(fc_feat).float()
            att_feat = torch.from_numpy(att_feat).float()
            
            if len(att_feat.shape) == 3:
                H, W, C = att_feat.shape
                att_feat = att_feat.reshape(H*W, C)
            
            label = torch.from_numpy(label).long()
            mask = torch.from_numpy(mask).float()
            
            return fc_feat, att_feat, label, mask
            
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {str(e)}")
            return self._get_default_item()
    
    def _get_default_item(self):
        """返回默认的数据项，而不是返回None"""
        # 创建默认的特征和标签
        fc_feat = torch.zeros(self.opt.fc_feat_size, dtype=torch.float32)
        att_feat = torch.zeros(196, self.opt.att_feat_size, dtype=torch.float32)  # 196 = 14*14
        label = torch.zeros(self.seq_length, dtype=torch.int64)
        mask = torch.zeros(self.seq_length, dtype=torch.float32)
        return fc_feat, att_feat, label, mask
    
    def _validate_shape(self, tensor, expected_shape):
        """验证tensor的维度是否符合预期"""
        if len(tensor.shape) != len(expected_shape):
            return False
        
        for actual, expected in zip(tensor.shape, expected_shape):
            if expected != -1 and actual != expected:
                return False
        
        return True
    
    def __del__(self):
        if self._label_file is not None:
            self._label_file.close()
            self._label_file = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_label_file'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

class BatchSizeManager:
    def __init__(self, initial_size=32, max_size=256, growth_factor=1.2, memory_threshold=0.8):
        self.current_size = initial_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.memory_threshold = memory_threshold
        self.stable_steps = 0
        
    def update(self, device):
        if device.type != 'cuda':
            return self.current_size
            
        # 获取当前GPU内存使用情况
        memory_used = torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
        
        if memory_used < self.memory_threshold:
            self.stable_steps += 1
            if self.stable_steps >= 50:  # 连续50步内存使用稳定
                new_size = min(int(self.current_size * self.growth_factor), self.max_size)
                if new_size != self.current_size:
                    print(f"\nIncreasing batch size from {self.current_size} to {new_size}")
                    self.current_size = new_size
                self.stable_steps = 0
        else:
            # 如果内存使用过高，减小批量大小
            self.current_size = max(16, int(self.current_size * 0.8))
            self.stable_steps = 0
            
        return self.current_size

def create_model(opt, device):
    """创建并初始化模型"""
    try:
        # 加载infos
        with open(opt.infos_path, 'rb') as f:
            infos = utils.pickle_load(f)
        
        # 处理参数
        if len(opt.id) == 0:
            opt.id = infos['opt'].id
        
        replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
        ignore = ['start_from', 'language_eval', 'dump_images', 'dump_json', 'dump_path', 'prune_amount', 'distill_epochs']

        for k in vars(infos['opt']).keys():
            if k not in ignore:
                if k in replace:
                    if not vars(opt)[k]:
                        vars(opt).update({k: vars(infos['opt'])[k]})
                else:
                    if k not in vars(opt):
                        vars(opt).update({k: vars(infos['opt'])[k]})
        
        opt.vocab = infos['vocab']
        
        # 初始化模型
        model = models.setup(opt)
        
        # 加载权重时指定设备
        state_dict = torch.load(
            os.path.join(opt.model_path, 'cap_model-best.pth'),
            map_location=device
        )
        model.load_state_dict(state_dict)
        
        return model
        
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

def move_to_device(model, inputs, device):
    """确保模型和输入在同一设备上"""
    model = model.to(device)
    
    if isinstance(inputs, (list, tuple)):
        return [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
    elif isinstance(inputs, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    return inputs

def create_dataloader(opt, batch_size=None):
    """创建数据加载器"""
    try:
        with open(opt.infos_path, 'rb') as f:
            infos = utils.pickle_load(f)
        
        dataset = CaptionDataset(opt, infos, 'train')
        
        # 使用传入的batch_size或默认值
        actual_batch_size = batch_size or opt.batch_size or 32
        
        loader = DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=4,  # 增加工作进程数
            pin_memory=True,
            prefetch_factor=3,  # 增加预取因子
            persistent_workers=True,
            drop_last=True
        )
        
        return loader, dataset
        
    except Exception as e:
        print(f"Error creating dataloader: {str(e)}")
        raise

def save_models(quantized_model, pruned_model, distilled_model, opt):
    """保存优化后的模型"""
    output_dir = os.path.join(opt.model_path, 'optimized_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    if quantized_model is not None:
        torch.save(quantized_model.state_dict(), 
                  os.path.join(output_dir, 'cap_model_quantized.pth'))
    
    if pruned_model is not None:
        torch.save(pruned_model.state_dict(), 
                  os.path.join(output_dir, 'cap_model_pruned.pth'))
    
    if distilled_model is not None:
        torch.save(distilled_model.state_dict(), 
                  os.path.join(output_dir, 'cap_model_distilled.pth'))
    
    # 保存优化信息
    optimization_info = {
        'original_size': utils.get_model_size(opt.original_model),
        'quantized_size': utils.get_model_size(quantized_model) if quantized_model is not None else None,
        'pruned_size': utils.get_model_size(pruned_model) if pruned_model is not None else None,
        'distilled_size': utils.get_model_size(distilled_model) if distilled_model is not None else None,
        'optimization_params': {
            'prune_amount': opt.prune_amount,
            'distill_epochs': opt.distill_epochs
        }
    }
    
    with open(os.path.join(output_dir, 'optimization_info.json'), 'w') as f:
        json.dump(optimization_info, f, indent=2)
    
    print("\nOptimization completed. Models saved in:", output_dir)

def verify_tensorboard_logs(log_dir):
    """验证 TensorBoard 日志文件是否正确生成"""
    try:
        event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.')]
        if not event_files:
            print(f"Warning: No TensorBoard event files found in {log_dir}")
            return False
            
        print(f"Found TensorBoard event files: {event_files}")
        return True
        
    except Exception as e:
        print(f"Error checking TensorBoard logs: {str(e)}")
        return False

def distill_model(teacher_model, student_model, train_loader, device, opt):
    """知识蒸馏训练过程"""
    # 确保模型在正确的设备上
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # 创建日志目录
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(opt.model_path, 'runs', f'distillation_{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        writer = SummaryWriter(
            log_dir=log_dir,
            flush_secs=1,
            max_queue=1
        )
        print(f"TensorBoard logs will be saved to: {log_dir}")
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        writer.add_text('setup', 'Training started', 0)
        writer.flush()
        
        writer.add_text('training_config', 
            f'batch_size: {opt.batch_size}\n'
            f'learning_rate: {opt.learning_rate}\n'
            f'temperature: {opt.temperature}\n'
            f'alpha: {opt.alpha}'
        )
        
        optimizer = optim.Adam(student_model.parameters(), lr=opt.learning_rate)
        ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
        mse_loss = nn.MSELoss().to(device)
        
        global_step = 0
        best_loss = float('inf')
        
        # 根据设备选择是否使用混合精度训练
        use_amp = device.type == 'cuda' and torch.cuda.is_available()
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        vocab_size = opt.vocab_size
        
        if not verify_tensorboard_logs(log_dir):
            print("Warning: TensorBoard logs may not be working correctly")
        
        # 修改检查点加载逻辑
        checkpoint_dir = os.path.join(opt.model_path, 'checkpoints')
        last_checkpoint = find_latest_checkpoint(checkpoint_dir)
        
        if last_checkpoint:
            try:
                state = load_checkpoint(last_checkpoint)
                # 尝试加载检查点
                student_model.load_state_dict(state['model_state_dict'])
                optimizer.load_state_dict(state['optimizer_state_dict'])
                start_epoch = state['epoch']
                global_step = state['global_step']
                best_loss = state['best_loss']
                torch.set_rng_state(state['rng_state'])
                print("Successfully loaded checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint due to {str(e)}")
                print("Starting training from scratch")
                # 如果加载失败，删除所有旧的检查点
                import shutil
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                os.makedirs(checkpoint_dir, exist_ok=True)
                start_epoch = 0
                global_step = 0
                best_loss = float('inf')
        else:
            print("No checkpoint found, starting from scratch")
            start_epoch = 0
            global_step = 0
            best_loss = float('inf')
        
        # 修改损失权重
        caption_weight = opt.alpha
        encoder_weight = (1 - opt.alpha) * 0.3  # 调整权重分配
        extractor_weight = (1 - opt.alpha) * 0.7  # 增加提取器的权重
        
        # 添加额外的正则化损失
        def compute_extractor_loss(student_extracted, teacher_extracted):
            # 基础 MSE 损失
            base_loss = mse_loss(student_extracted, teacher_extracted)
            
            # 添加 L1 正则化，促进稀疏性
            l1_reg = 0.01 * torch.mean(torch.abs(student_extracted))
            
            # 添加 KL 散度损失，确保分布匹配
            kl_div = F.kl_div(
                F.log_softmax(student_extracted, dim=-1),
                F.softmax(teacher_extracted, dim=-1),
                reduction='batchmean'
            )
            
            return base_loss + l1_reg + 0.1 * kl_div
        
        # 添加预热和学习率调度
        warmup_steps = 1000
        total_steps = opt.distill_epochs * len(train_loader)
        
        def get_lr_scale(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.1 ** (step / (total_steps / 3))
        
        # 创建批量大小管理器
        batch_manager = BatchSizeManager(
            initial_size=opt.batch_size or 32,
            max_size=256,
            growth_factor=1.2,
            memory_threshold=0.8
        )
        
        dataset = train_loader.dataset
        
        # 训练循环中
        try:
            for epoch in range(start_epoch, opt.distill_epochs):
                # 每个epoch开始时更新批量大小
                new_batch_size = batch_manager.update(device)
                if new_batch_size != train_loader.batch_size:
                    train_loader = DataLoader(
                        dataset,
                        batch_size=new_batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        prefetch_factor=3,
                        persistent_workers=True,
                        drop_last=True
                    )
                    print(f"Epoch {epoch}: Updated batch size to {new_batch_size}")
                
                # 每个epoch结束或遇到错误时保存检查点
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'rng_state': torch.get_rng_state(),
                }, checkpoint_dir)
                
                epoch_loss = 0
                epoch_distill_loss = 0
                epoch_label_loss = 0
                n_batches = 0
                
                pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
                teacher_model.eval()
                student_model.train()
                
                for i, batch in enumerate(pbar):
                    try:
                        if batch is None:
                            continue
                            
                        if i % 50 == 0:
                            torch.cuda.empty_cache() if device.type == 'cuda' else None
                        
                        optimizer.zero_grad(set_to_none=True)
                        
                        # 将数据移到设备上
                        fc_feats, att_feats, labels, masks = [
                            x.to(device, non_blocking=True) if x is not None else None 
                            for x in batch
                        ]
                        
                        # 生成随机秘密信息
                        secret = torch.randn(fc_feats.size(0), opt.secret_size).to(device)
                        
                        with torch.no_grad():
                            # 1. 教师模型的图像描述生成
                            mean_feats, att_feats_prepared, p_att_feats, _ = teacher_model._prepare_feature(
                                fc_feats, att_feats, masks)
                            
                            state = teacher_model.init_hidden(fc_feats.size(0))
                            if isinstance(state, tuple):
                                state = tuple(s.to(device) for s in state)
                            else:
                                state = state.to(device)
                            
                            seq = labels[:, 0]
                            xt = teacher_model.embed(seq)
                            
                            output, _ = teacher_model.core(
                                xt,
                                mean_feats,
                                att_feats_prepared,
                                p_att_feats,
                                state,
                                masks
                            )
                            
                            teacher_logits = teacher_model.logit(output)
                            teacher_logits = teacher_logits[:, :vocab_size]
                            teacher_probs = F.softmax(teacher_logits / opt.temperature, dim=-1)
                            
                            # 2. 获取教师模型的秘密信息编码
                            # 创建并加载秘密编码器
                            teacher_sec_encoder = SecretEncoder(opt.secret_size).to(device)
                            teacher_sec_encoder.load_state_dict(
                                torch.load(os.path.join(opt.model_path, 'sec_encoder-best.pth'))
                            )
                            teacher_sec_encoder.eval()
                            
                            # 创建并加载秘密提取器
                            teacher_sec_extractor = SecretExtractor(opt).to(device)
                            teacher_sec_extractor.load_state_dict(
                                torch.load(os.path.join(opt.model_path, 'sec_extractor-best.pth'))
                            )
                            teacher_sec_extractor.eval()
                            
                            # 获取教师模型的秘密信息处理结果
                            teacher_encoded = teacher_sec_encoder(secret)  # [B, 196, 2048]
                            
                            # 使用教师模型生成序列和密钥
                            teacher_seq, teacher_key, _ = teacher_model(fc_feats, att_feats, masks, mode='sample')
                            teacher_extracted = teacher_sec_extractor(teacher_seq, teacher_key)
                        
                        # 反向传播部分需要修改
                        if use_amp:
                            # 使用混合精度训练
                            with torch.cuda.amp.autocast():
                                # 1. 获取 logits 用于描述生成
                                student_logits = student_model(fc_feats, att_feats, masks, mode='forward')
                                
                                # 2. 获取序列和密钥用于秘密信息提取
                                student_seq, student_key, _ = student_model(fc_feats, att_feats, masks, mode='sample')
                                
                                # 3. 编码和提取秘密信息
                                student_encoded = student_model.encode_secret(secret)
                                student_extracted = student_model.extract_secret(student_seq, student_key)
                                
                                # 计算损失
                                # 1. 描述生成损失
                                caption_loss = kl_loss(
                                    F.log_softmax(student_logits / opt.temperature, dim=-1),
                                    teacher_probs
                                ) * (opt.temperature ** 2)
                                
                                # 2. 秘密信息编码损失
                                encoder_loss = mse_loss(student_encoded, teacher_encoded)
                                
                                # 3. 秘密信息提取损失
                                extractor_loss = compute_extractor_loss(student_extracted, teacher_extracted)
                                
                                # 总损失
                                loss = (caption_weight * caption_loss + 
                                       encoder_weight * encoder_loss + 
                                       extractor_weight * extractor_loss)
                            
                            # 使用 scaler 进行反向传播
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # 普通训练
                            # 1. 获取 logits 用于描述生成
                            student_logits = student_model(fc_feats, att_feats, masks, mode='forward')
                            
                            # 2. 获取序列和密钥用于秘密信息提取
                            student_seq, student_key, _ = student_model(fc_feats, att_feats, masks, mode='sample')
                            
                            # 3. 编码和提取秘密信息
                            student_encoded = student_model.encode_secret(secret)
                            student_extracted = student_model.extract_secret(student_seq, student_key)
                            
                            # 计算损失
                            # 1. 描述生成损失
                            caption_loss = kl_loss(
                                F.log_softmax(student_logits / opt.temperature, dim=-1),
                                teacher_probs
                            ) * (opt.temperature ** 2)
                            
                            # 2. 秘密信息编码损失
                            encoder_loss = mse_loss(student_encoded, teacher_encoded)
                            
                            # 3. 秘密信息提取损失
                            extractor_loss = compute_extractor_loss(student_extracted, teacher_extracted)
                            
                            # 总损失
                            loss = (caption_weight * caption_loss + 
                                   encoder_weight * encoder_loss + 
                                   extractor_weight * extractor_loss)
                            
                            # 普通反向传播
                            loss.backward()
                            optimizer.step()
                        
                        # 更新统计信息
                        epoch_loss += loss.item()
                        epoch_distill_loss += caption_loss.item()
                        epoch_label_loss += encoder_loss.item() + extractor_loss.item()
                        n_batches += 1
                        
                        if i % 5 == 0:
                            step = epoch * len(train_loader) + i
                            writer.add_scalar('loss/total', loss.item(), step)
                            writer.add_scalar('loss/distill', caption_loss.item(), step)
                            writer.add_scalar('loss/label', encoder_loss.item() + extractor_loss.item(), step)
                            writer.add_scalar('memory/used_mb', 
                                torch.cuda.memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0, 
                                step)
                            writer.flush()
                        global_step += 1
                        
                        if i % 10 == 0:
                            if device.type == 'cuda':
                                memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                                memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
                            else:
                                memory_allocated = 0
                                memory_reserved = 0
                                
                            pbar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'memory_used': f'{memory_allocated:.0f}MB',
                                'memory_reserved': f'{memory_reserved:.0f}MB'
                            })
                        
                        del output, teacher_logits
                        if i % 2 == 0 and device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        # 记录所有损失
                        writer.add_scalar('loss/caption', caption_loss.item(), global_step)
                        writer.add_scalar('loss/encoder', encoder_loss.item(), global_step)
                        writer.add_scalar('loss/extractor', extractor_loss.item(), global_step)
                        
                        # 添加梯度裁剪
                        if not use_amp:
                            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                        
                        # 添加学习率调度
                        if i % 1000 == 0:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = opt.learning_rate * (0.1 ** (epoch // 2))
                        
                    except Exception as e:
                        print(f"Error in batch {i}: {str(e)}")
                        print(f"Device info - teacher: {next(teacher_model.parameters()).device}, "
                              f"student: {next(student_model.parameters()).device}")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                
                # 每个epoch结束时记录
                avg_loss = epoch_loss / n_batches
                writer.add_scalar('epoch/average_loss', avg_loss, epoch)
                writer.flush()
                
                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(opt.model_path, 'best_student_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, save_path)
                    writer.add_text('checkpoint', 
                        f'Epoch {epoch}: Saved model with loss {best_loss:.4f}')
                
                # 每个epoch结束时清理
                torch.cuda.empty_cache()
            
        except Exception as e:
            # 保存最后状态
            save_checkpoint({...}, checkpoint_dir, is_error=True)
            raise e
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise
    finally:
        writer.flush()
        writer.close()
    
    return student_model

def quantize_model(model):
    """对模型进行动态量化"""
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized_model

def optimize_model():
    """主函数 - 单GPU版本"""
    try:
        # 使用指定的GPU
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # 使用显存较大的GPU
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
        
        print("1. Loading teacher model...")
        teacher_model = create_model(opt, device)
        teacher_model.eval()
        
        print("2. Creating student model...")
        student_model = SimpleStudentModel(
            fc_feat_size=opt.fc_feat_size,
            att_feat_size=opt.att_feat_size,
            hidden_size=opt.hidden_size,
            vocab_size=opt.vocab_size,
            secret_size=opt.secret_size
        ).to(device)
        
        print("3. Creating data loader...")
        train_loader, dataset = create_dataloader(opt)
        
        print("4. Starting knowledge distillation...")
        distilled_model = distill_model(
            teacher_model,
            student_model,
            train_loader,
            device,
            opt
        )
        
        print("5. Applying quantization...")
        quantized_model = quantize_model(distilled_model.cpu())
        
        print("6. Saving optimized model...")
        save_dir = os.path.join(opt.model_path, 'optimized_models')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(quantized_model.state_dict(), 
                  os.path.join(save_dir, 'quantized_student_model.pth'))
        print(f"Model saved to {save_dir}")
        print("Optimization completed successfully!")
        
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--model_path', type=str, default='',
                    help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
    parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
    parser.add_argument('--input_json', type=str, default='',
                    help='path to the input json file')
    parser.add_argument('--input_fc_dir', type=str, default='',
                    help='path to the input fc feats file')
    parser.add_argument('--input_att_dir', type=str, default='',
                    help='path to the input att feats file')
    parser.add_argument('--input_box_dir', type=str, default='',
                    help='path to the input box feats file')
    parser.add_argument('--input_label_h5', type=str, default='',
                    help='path to the input labels file')

    # 模型参数 (合并所有模型相关参数)
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='Size of fc features')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='Size of attention features')
    parser.add_argument('--hidden_size', type=int, default=512,
                    help='Hidden size for student model')
    parser.add_argument('--vocab_size', type=int, default=9487,
                    help='Size of vocabulary')
    parser.add_argument('--secret_size', type=int, default=10,
                    help='Size of secret information')

    # 优化相关参数
    parser.add_argument('--prune_amount', type=float, default=0.3,
                    help='Amount of parameters to prune')
    parser.add_argument('--distill_epochs', type=int, default=5,
                    help='Number of epochs for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight for distillation loss')
    parser.add_argument('--temperature', type=float, default=2.0,
                    help='Temperature for knowledge distillation')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Learning rate for distillation')
    
    # Basic options
    parser.add_argument('--batch_size', type=int, default=0,
                    help='if > 0 then overrule, otherwise load from checkpoint.')
    parser.add_argument('--num_images', type=int, default=-1,
                    help='how many images to use when periodically evaluating the loss? (-1 = all)')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)?')
    parser.add_argument('--dump_images', type=int, default=1,
                    help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
    parser.add_argument('--dump_json', type=int, default=1,
                    help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--dump_path', type=int, default=0,
                    help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job.')
    
    opt = parser.parse_args()
    
    # 添加参数验证
    required_params = ['fc_feat_size', 'att_feat_size', 'hidden_size', 
                      'vocab_size', 'secret_size']
    for param in required_params:
        if not hasattr(opt, param):
            raise ValueError(f"Missing required parameter: {param}")
    
    try:
        optimize_model()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")