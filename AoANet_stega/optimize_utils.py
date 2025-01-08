import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import glob
import numpy as np

def move_to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [move_to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    return data

class SecretExtractorModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, secret_size):
        super().__init__()
        
        # 嵌入层
        self.sec_embed = nn.Embedding(vocab_size + 1, hidden_size)
        self.key_embed = nn.Linear(vocab_size + 1, hidden_size)
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, 4)
        
        # 输出层
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, secret_size)
        )
        
    def forward(self, seq, key):
        # 1. 嵌入处理
        seq_emb = self.sec_embed(seq)  # [B, hidden_size]
        key_emb = self.key_embed(key)  # [B, hidden_size]
        
        # 2. 特征融合
        combined = torch.cat([seq_emb, key_emb], dim=-1)
        features = self.feature_net(combined)
        
        # 3. 自注意力处理
        features = features.unsqueeze(0)  # [1, B, H]
        attn_out, _ = self.attention(features, features, features)
        features = attn_out.squeeze(0)  # [B, H]
        
        # 4. 输出处理
        secret = self.output_net(features)
        return torch.sigmoid(secret)

class SimpleStudentModel(nn.Module):
    """简化的学生模型，整合了所有功能"""
    def __init__(self, fc_feat_size, att_feat_size, hidden_size, vocab_size, secret_size):
        super(SimpleStudentModel, self).__init__()
        
        # 特征处理部分 (对应 AoAModel)
        self.fc_embed = nn.Linear(fc_feat_size, hidden_size)
        self.att_embed = nn.Linear(att_feat_size, hidden_size)
        self.logit = nn.Linear(hidden_size, vocab_size)
        
        # 秘密信息编码部分 (完全匹配 SecretEncoder)
        self.enc_linear1 = nn.Linear(secret_size, 2048)
        self.enc_linear2 = nn.Linear(1, 196)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.enc_linear1.weight)
        nn.init.kaiming_normal_(self.enc_linear2.weight)
        
        # 秘密信息提取部分 (完全匹配 SecretExtractor)
        self.secret_extractor = SecretExtractorModule(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            secret_size=secret_size
        )
        
        self.dropout = nn.Dropout(0.5)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
    
    def encode_secret(self, secret):
        """秘密信息编码，匹配原始 SecretEncoder"""
        x = self.enc_linear1(secret)  # [B, 2048]
        x = x.unsqueeze(2)  # [B, 2048, 1]
        x = self.enc_linear2(x)  # [B, 2048, 196]
        x = self.relu(x.permute(0, 2, 1))  # [B, 196, 2048]
        return x
    
    def extract_secret(self, sec_seq, key):
        """秘密信息提取，使用 SecretExtractorModule"""
        # 直接使用 secret_extractor 模块
        return self.secret_extractor(sec_seq, key)
    
    def forward(self, fc_feats, att_feats, att_masks=None, mode='forward'):
        """完全匹配原始模型的前向传播"""
        # 特征处理
        fc_embed = self.dropout(self.fc_embed(fc_feats))
        att_embed = self.dropout(self.att_embed(att_feats))
        
        # 融合特征
        fused_features = fc_embed + att_embed.mean(1)
        
        # 生成描述的logits
        logits = self.logit(fused_features)
        
        if mode == 'sample':
            # 生成序列和密钥
            seq = torch.argmax(logits, dim=-1)  # [B]
            key = torch.zeros(fc_feats.size(0), self.vocab_size + 1).to(fc_feats.device)  # [B, vocab_size+1]
            return seq, key, None
        
        return logits

def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return None
        
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth'))
    if not checkpoints:
        return None
    
    # 修改检查点文件筛选逻辑
    valid_checkpoints = []
    for ckpt in checkpoints:
        try:
            # 尝试从文件名中提取 epoch 数字
            epoch = int(os.path.basename(ckpt).split('_')[1].split('.')[0])
            valid_checkpoints.append((ckpt, epoch))
        except (ValueError, IndexError):
            # 跳过无法解析的文件名
            continue
    
    if not valid_checkpoints:
        return None
        
    # 按 epoch 数排序并返回最新的
    latest_checkpoint = max(valid_checkpoints, key=lambda x: x[1])[0]
    return latest_checkpoint

def save_checkpoint(state, checkpoint_dir, is_error=False):
    """保存检查点"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 修改文件命名逻辑
        if is_error:
            filename = 'checkpoint_backup.pth'  # 改为更通用的名称
        else:
            filename = f'checkpoint_{state["epoch"]:03d}.pth'
            
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(state, filepath)
        
        print(f"Checkpoint saved: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")
        return None

def load_checkpoint(checkpoint_path):
    """加载检查点"""
    try:
        state = torch.load(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Resuming from epoch {state['epoch']}")
        return state
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None

class FeatureLoader:
    def __init__(self, input_dir, input_type, max_cache_size=100):
        self.input_dir = input_dir
        self.input_type = input_type
        self.handlers = {}
        self.max_cache_size = max_cache_size
        
    def get(self, key):
        if key in self.handlers:
            return self.handlers[key]
            
        # 如果缓存太大，清理一些
        if len(self.handlers) > self.max_cache_size:
            # 删除最早的一半缓存
            remove_keys = list(self.handlers.keys())[:len(self.handlers)//2]
            for k in remove_keys:
                del self.handlers[k]
        
        # 加载新特征
        feat_path = os.path.join(self.input_dir, str(key))
        feat = np.load(feat_path + ('.npy' if self.input_type == 'fc' else '.npz')['feat'])
        
        self.handlers[key] = feat
        return feat