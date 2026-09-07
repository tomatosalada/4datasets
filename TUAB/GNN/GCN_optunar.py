import mne
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, DenseGATConv
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

#from GNN.GNN_set import edge_index

import math

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=2, concat=True, dropout=0.30701473066919766):
        super(GraphTransformerLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # --- 1. GATv2 Attention Part ---
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_features))
        
        # Scaling factor (学習安定化)
        self.scale = 1.0 / math.sqrt(out_features)

        # Learnable Bias (固定グラフの補正用: 16ノード固定)
        # 任意のつながりを学習できるようにする
        self.bias = nn.Parameter(torch.zeros(1, heads, 16, 16)) 

        if concat:
            self.norm1 = nn.LayerNorm(heads * out_features)
        else:
            self.norm1 = nn.LayerNorm(out_features)

        # Skip connection projection
        hidden_dim = heads * out_features if concat else out_features
        if in_features != hidden_dim:
            self.skip_proj1 = nn.Linear(in_features, hidden_dim)
        else:
            self.skip_proj1 = None

        # --- 2. Feed-Forward Network (FFN) Part ---
        # これを追加することで表現力が爆発的に上がります
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), # 中間層を広げるのが一般的
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # 初期化
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_normal_(self.bias) # バイアスは小さく初期化
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()
        
        # === Block 1: Graph Attention ===
        residual = x
        if self.skip_proj1 is not None:
            residual = self.skip_proj1(residual)

        # Linear Transform
        x_feat = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_features)
        
        # GATv2 Score Calculation
        x_i = x_feat.unsqueeze(2)
        x_j = x_feat.unsqueeze(1)
        
        x_pair = self.leaky_relu(x_i + x_j)
        score = (x_pair * self.att.view(1, 1, 1, self.heads, self.out_features)).sum(dim=-1)
        
        # Scaling & Bias Addition
        score = score * self.scale # スケール調整
        score = score.permute(0, 3, 1, 2) # [B, H, N, N]
        
        # Bias加算 (固定グラフ + 学習可能バイアス)
        # これにより「固定グラフにない重要な結合」も見つけられる
        score = score + self.bias 
        
        # Masking
        adj = adj.unsqueeze(1)
        zero_vec = -9e15 * torch.ones_like(score)
        # adj>0 の場所は scoreを採用、それ以外も bias があるので完全には切らない方が良いが、
        # ここでは元のGATの思想を守りつつ、つながっている部分の重みを調整する方針にする
        attention = torch.where(adj > 0, score, zero_vec)
        
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregation
        x_feat = x_feat.permute(0, 2, 1, 3)
        out = torch.matmul(attention, x_feat)
        
        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)
            
        # Residual & Norm 1
        out = out + residual
        out = self.norm1(out)

        # === Block 2: Feed-Forward Network (FFN) ===
        # ここでノードごとの特徴をさらに深掘りする
        residual = out
        out = self.ffn(out)
        out = out + residual
        out = self.norm2(out)

        return out, attention

class DenseGAT3(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2, dropout=0.30701473066919766):
        super().__init__()
        head_dim = out_channels // heads
        
        # GraphTransformerLayer に置き換え
        self.conv1 = GraphTransformerLayer(in_channels, head_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GraphTransformerLayer(out_channels, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, adj):
        x, atten1 = self.conv1(x, adj)
        x, _ = self.conv2(x, adj)
        return x, atten1