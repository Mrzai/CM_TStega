import torch
import numpy as np

def get_data_iterator(loader):
    """创建一个数据迭代器，只返回需要的字段"""
    for data in loader:
        try:
            if len(data) == 6:  # 原始数据格式
                fc_feats, att_feats, _, _, att_masks, _ = data
            elif len(data) == 4:  # 简化的数据格式
                fc_feats, att_feats, _, att_masks = data
            else:
                raise ValueError(f"Unexpected data format with {len(data)} elements")
            
            # 转换numpy数组为torch张量
            if isinstance(fc_feats, np.ndarray):
                fc_feats = torch.from_numpy(fc_feats)
            if isinstance(att_feats, np.ndarray):
                att_feats = torch.from_numpy(att_feats)
            
            # 特殊处理att_masks
            if isinstance(att_masks, (int, np.int32, np.int64)):
                att_masks = torch.ones(att_feats.shape[:-1])  # 使用att_feats的形状但去掉最后一维
            elif isinstance(att_masks, np.ndarray):
                att_masks = torch.from_numpy(att_masks)
            elif not isinstance(att_masks, torch.Tensor):
                att_masks = torch.tensor(att_masks)
            
            # 确保维度正确
            if len(fc_feats.shape) == 1:
                fc_feats = fc_feats.unsqueeze(0)
            if len(att_feats.shape) == 2:
                att_feats = att_feats.unsqueeze(0)
            if len(att_masks.shape) == 1:
                att_masks = att_masks.unsqueeze(0)
            
            # 确保数据类型正确
            fc_feats = fc_feats.float()
            att_feats = att_feats.float()
            att_masks = att_masks.long()
            
            yield fc_feats, att_feats, att_masks
            
        except Exception as e:
            print(f"\nData iterator error: {str(e)}")
            print(f"Data format: {[d.shape if hasattr(d, 'shape') else type(d) for d in data]}")
            print(f"Data types: {[type(d) for d in data]}")
            if 'fc_feats' in locals():
                print(f"fc_feats shape: {fc_feats.shape}, type: {type(fc_feats)}")
            if 'att_feats' in locals():
                print(f"att_feats shape: {att_feats.shape}, type: {type(att_feats)}")
            if 'att_masks' in locals():
                print(f"att_masks shape: {att_masks.shape}, type: {type(att_masks)}")
            raise 