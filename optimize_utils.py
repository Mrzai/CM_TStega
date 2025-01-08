import torch
import torch.nn as nn
import torch.quantization
import copy
import numpy as np
from torch.nn.utils import prune

class ModelOptimizer:
    def __init__(self, original_model, calibration_loader):
        self.original_model = original_model
        self.calibration_loader = calibration_loader
        self.optimized_model = None
        
    def _process_batch(self, data):
        """统一处理数据批次"""
        try:
            if isinstance(data, (list, tuple)):
                if len(data) == 6:
                    fc_feats, att_feats, _, _, att_masks, _ = data
                elif len(data) == 4:
                    fc_feats, att_feats, _, att_masks = data
                elif len(data) == 3:
                    fc_feats, att_feats, att_masks = data
                else:
                    raise ValueError(f"Unexpected data format with {len(data)} elements")
            else:
                raise ValueError(f"Unexpected data type: {type(data)}")
            
            # 打印原始形状
            print("\nOriginal shapes:")
            print(f"fc_feats: {fc_feats.shape if hasattr(fc_feats, 'shape') else None}")
            print(f"att_feats: {att_feats.shape if hasattr(att_feats, 'shape') else None}")
            print(f"att_masks: {att_masks if isinstance(att_masks, (int, np.int32, np.int64)) else att_masks.shape if hasattr(att_masks, 'shape') else None}")
            
            # 转换numpy数组为torch张量
            if isinstance(fc_feats, np.ndarray):
                fc_feats = torch.from_numpy(fc_feats)
            if isinstance(att_feats, np.ndarray):
                att_feats = torch.from_numpy(att_feats)
            
            # 特殊处理att_masks
            if isinstance(att_masks, (int, np.int32, np.int64)):
                # 如果att_masks是整数，创建相应大小的mask
                att_masks = torch.ones(att_feats.shape[:-1])  # 使用att_feats的形状但去掉最后一维
            elif isinstance(att_masks, np.ndarray):
                att_masks = torch.from_numpy(att_masks)
            elif not isinstance(att_masks, torch.Tensor):
                att_masks = torch.tensor(att_masks)
            
            # 确保维度正确
            if len(fc_feats.shape) == 1:
                fc_feats = fc_feats.unsqueeze(0)  # [D] -> [1, D]
            if len(att_feats.shape) == 2:
                att_feats = att_feats.unsqueeze(0)  # [L, D] -> [1, L, D]
            if len(att_masks.shape) == 1:
                att_masks = att_masks.unsqueeze(0)  # [L] -> [1, L]
                
            # 确保数据类型正确
            fc_feats = fc_feats.float()
            att_feats = att_feats.float()
            att_masks = att_masks.long()  # 修改为长整型
            
            # 打印处理后的形状
            print("\nProcessed shapes:")
            print(f"fc_feats: {fc_feats.shape}")
            print(f"att_feats: {att_feats.shape}")
            print(f"att_masks: {att_masks.shape}")
            
            # 添加内存优化
            @torch.cuda.amp.autocast()  # 使用自动混合精度
            def process_tensors():
                nonlocal fc_feats, att_feats, att_masks
                # 移动到GPU并确保类型正确
                fc_feats = fc_feats.cuda().float()
                att_feats = att_feats.cuda().float()
                att_masks = att_masks.cuda().long()
                return fc_feats, att_feats, att_masks
            
            return process_tensors()
        except Exception as e:
            print(f"\nData processing error: {str(e)}")
            print(f"Data format: {[type(d) for d in data if isinstance(data, (list, tuple)) else type(data)]}")
            if 'fc_feats' in locals():
                print(f"fc_feats shape: {fc_feats.shape if hasattr(fc_feats, 'shape') else None}, type: {type(fc_feats)}")
            if 'att_feats' in locals():
                print(f"att_feats shape: {att_feats.shape if hasattr(att_feats, 'shape') else None}, type: {type(att_feats)}")
            if 'att_masks' in locals():
                print(f"att_masks shape: {att_masks.shape if hasattr(att_masks, 'shape') else None}, type: {type(att_masks)}")
            raise
        
    def quantize(self):
        """量化模型"""
        model = copy.deepcopy(self.original_model)
        model.eval()
        
        # 配置量化参数
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 为量化准备模型
        model_prepared = torch.quantization.prepare(model)
        
        # 校准
        print("Calibrating...")
        with torch.no_grad():
            try:
                for i, data in enumerate(self.calibration_loader):
                    if i > 100:  # 使用有限的样本进行校准
                        break
                    fc_feats, att_feats, att_masks = self._process_batch(data)
                    output = model_prepared(fc_feats, att_feats, att_masks)
                    # 确保输出被使用，避免被优化掉
                    if isinstance(output, tuple):
                        for o in output:
                            if hasattr(o, 'numel'):
                                _ = o.numel()
                    else:
                        _ = output.numel()
            except Exception as e:
                print(f"Calibration error at batch {i}: {str(e)}")
                raise
                
        # 转换为量化模型
        print("Converting to quantized model...")
        self.optimized_model = torch.quantization.convert(model_prepared)
        
        return self.optimized_model
        
    def prune(self, amount=0.3):
        """剪枝模型"""
        model = copy.deepcopy(self.original_model if self.optimized_model is None 
                            else self.optimized_model)
        
        # 对不同类型的层应用不同的剪枝策略
        for name, module in model.named_modules():
            # 线性层的剪枝
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 
                                    name='weight',
                                    amount=amount)
                
            # LSTM层的剪枝
            elif isinstance(module, nn.LSTM):
                prune.l1_unstructured(module, 
                                    name='weight_ih_l0',
                                    amount=amount)
                prune.l1_unstructured(module, 
                                    name='weight_hh_l0',
                                    amount=amount)
                
        # 使剪枝永久化
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LSTM)):
                for k, v in module.named_parameters():
                    if k == 'weight_orig':
                        prune.remove(module, 'weight')
                        
        self.optimized_model = model
        return self.optimized_model
        
    def distill(self, train_loader, epochs=5, batch_size=4):
        """知识蒸馏"""
        teacher = self.original_model
        student = copy.deepcopy(self.optimized_model if self.optimized_model 
                              else self.original_model)
        
        teacher.eval()
        student.train()
        
        # 配置蒸馏
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        temperature = 4.0
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        # 添加内存管理
        def clear_gpu_memory():
            torch.cuda.empty_cache()
        
        print("Starting knowledge distillation...")
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            accumulated_batches = []
            
            for i, data in enumerate(train_loader):
                if i > 100:  # 限制每个epoch的样本数
                    break
                    
                fc_feats, att_feats, att_masks = self._process_batch(data)
                accumulated_batches.append((fc_feats, att_feats, att_masks))
                
                if len(accumulated_batches) == batch_size:
                    # 处理累积的批次
                    batch_loss = 0
                    for batch_data in accumulated_batches:
                        b_fc_feats, b_att_feats, b_att_masks = batch_data
                        
                        # 教师模型输出
                        with torch.no_grad():
                            teacher_outputs = teacher(b_fc_feats, b_att_feats, b_att_masks)
                        
                        # 学生模型输出
                        student_outputs = student(b_fc_feats, b_att_feats, b_att_masks)
                        
                        # 计算蒸馏loss
                        distillation_loss = criterion(
                            torch.log_softmax(student_outputs/temperature, dim=1),
                            torch.softmax(teacher_outputs/temperature, dim=1)
                        ) * (temperature * temperature)
                        
                        batch_loss += distillation_loss.item()
                        distillation_loss.backward()
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += batch_loss
                    batch_count += 1
                    accumulated_batches = []
                    
                    # 清理GPU内存
                    clear_gpu_memory()
                
                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}, "
                          f"Loss: {total_loss/batch_count if batch_count > 0 else 0:.4f}")
            
            # 处理剩余的批次
            if accumulated_batches:
                # ... (类似上面的批次处理逻辑)
                pass
            
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Average Loss: {total_loss/batch_count if batch_count > 0 else 0:.4f}")
        
        self.optimized_model = student
        return self.optimized_model 