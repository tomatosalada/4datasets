import mne
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, DenseGATConv
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

#from GNN.GNN_set import edge_index

class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x

class GAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(GAT, self).__init__()


        self.dropout = nn.Dropout(p=0.3)
        self.conv1 = GATConv(in_features, 32, heads=8)
        self.norm1 = nn.LayerNorm(256)
        self.conv2 = GATConv(256, 16, heads=8)
        self.norm2 = nn.LayerNorm(128)
        self.conv3 = GATConv(128, out_features // 8, heads=8)
        self.norm3 = nn.LayerNorm(out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1, attention_weights_1 = self.conv1(x, edge_index, return_attention_weights=True)
        x1 = self.norm1(x1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training = self.training)
        x2, attention_weights_2 = self.conv2(x1, edge_index, return_attention_weights=True)
        x2 = self.norm2(x2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3, attention_weights_3 = self.conv3(x2, edge_index, return_attention_weights = True)
        x3 = self.norm3(x3)
        x3 = F.elu(x3)
        return x3, attention_weights_1, attention_weights_2, attention_weights_3

class DenseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.2):
        super(DenseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_features))
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()
        x_feat = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_features)
        
        alpha_src = (x_feat * self.att_src).sum(dim=-1)
        alpha_dst = (x_feat * self.att_dst).sum(dim=-1)
        
        score = alpha_src.view(batch_size, num_nodes, self.heads, 1) + \
                alpha_dst.view(batch_size, 1, self.heads, num_nodes)
        
        score = self.leaky_relu(score.permute(0, 2, 1, 3))
        
        adj = adj.unsqueeze(1)
        zero_vec = -9e15 * torch.ones_like(score)
        attention = torch.where(adj > 0, score, zero_vec)
        
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        x_feat = x_feat.permute(0, 2, 1, 3)
        out = torch.matmul(attention, x_feat)
        
        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        return out, attention

class DenseGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        # 入出力次元の調整
        head_dim = out_channels // heads
        
        # 第1層: ここでAttention (構造学習) を取得したい
        self.conv1 = DenseGATLayer(in_channels, head_dim, heads=heads, concat=True, dropout=dropout)
        # 第2層
        self.conv2 = DenseGATLayer(in_channels, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, adj):
        # atten1 を取得して返すように変更
        x, atten1 = self.conv1(x, adj)
        x = F.elu(x)
        x, _ = self.conv2(x, adj) # 第2層のAttentionは今回は無視
        return x, atten1

class ModernDenseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.2):
        super(ModernDenseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # --- GATv2用パラメータ ---
        # 入力を変換するLinear
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        
        # Attentionスコアを計算するベクトル 'a' (GATv2ではLinearの後、LeakyReLUの後にかける)
        self.att = nn.Parameter(torch.Tensor(1, heads, out_features))
        
        # --- 安定化のための追加 ---
        # Layer Normalization
        if concat:
            self.norm = nn.LayerNorm(heads * out_features)
        else:
            self.norm = nn.LayerNorm(out_features)
            
        # Skip Connection用の射影 (入力次元 != 出力次元の場合のみ使用)
        if in_features != (heads * out_features if concat else out_features):
            self.skip_proj = nn.Linear(in_features, (heads * out_features if concat else out_features))
        else:
            self.skip_proj = None

        # 初期化
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.size()
        
        # 1. 線形変換 [B, N, H, F_out]
        x_feat = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_features)
        
        # --- GATv2の実装 (Dynamic Attention) ---
        # GATv1: LeakyReLU(a * x_i + a * x_j)
        # GATv2: a * LeakyReLU(x_i + x_j)  <-- ここが違う！
        
        # x_i + x_j の全ペアを作成 (Broadcasting)
        # [B, N, 1, H, F] + [B, 1, N, H, F] = [B, N, N, H, F]
        # メモリ節約のため、計算順序を工夫します
        
        # まず x_feat を [B, N, 1, H, F] と [B, 1, N, H, F] に拡張して加算するイメージですが
        # ここではアインシュタイン総和規約っぽく計算する代わりに、
        # 前回のコードと同様の次元操作で行います。
        
        # x_feat: [B, N, H, F]
        # x_i: [B, N, 1, H, F]
        # x_j: [B, 1, N, H, F]
        # sum: [B, N, N, H, F] -> LeakyReLU -> dot(att) -> [B, N, N, H]
        
        # 注意: N=18程度ならこのまま展開してもGPUメモリは大丈夫です
        x_i = x_feat.unsqueeze(2) # [B, N, 1, H, F]
        x_j = x_feat.unsqueeze(1) # [B, 1, N, H, F]
        
        # 非線形変換を先に行う
        x_pair = self.leaky_relu(x_i + x_j) 
        
        # Attentionベクトルとの内積をとる
        # att: [1, H, F] -> [1, 1, 1, H, F]
        # sum over F dimension -> score: [B, N, N, H]
        score = (x_pair * self.att.view(1, 1, 1, self.heads, self.out_features)).sum(dim=-1)
        
        # [B, N, N, H] -> [B, H, N, N] に並べ替え
        score = score.permute(0, 3, 1, 2)
        
        # --- マスキングと正規化 ---
        adj = adj.unsqueeze(1) # [B, 1, N, N]
        zero_vec = -9e15 * torch.ones_like(score)
        attention = torch.where(adj > 0, score, zero_vec)
        
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # --- 集約 ---
        # x_feat: [B, N, H, F] -> [B, H, N, F]
        x_feat = x_feat.permute(0, 2, 1, 3)
        
        # attention: [B, H, N, N]
        # matmul -> [B, H, N, F]
        out = torch.matmul(attention, x_feat)
        
        # 整形
        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)
            
        # --- Skip Connection & LayerNorm ---
        # 入力 x を加算 (次元が違う場合は射影する)
        if self.skip_proj is not None:
            x = self.skip_proj(x)
            
        out = out + x  # Residual Connection
        out = self.norm(out) # Layer Normalization

        return out, attention

class DenseGAT2(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        head_dim = out_channels // heads
        
        # GATv2化したレイヤーを使用
        self.conv1 = ModernDenseGATLayer(in_channels, head_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = ModernDenseGATLayer(out_channels, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, adj):
        x, atten1 = self.conv1(x, adj)
        x = F.gelu(x) # ★変更: ELUよりGELUの方がTransformer系では性能が良い傾向
        x, _ = self.conv2(x, adj)
        return x, atten1

import math

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, concat=True, dropout=0.3):
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

        # Learnable Bias (固定グラフの補正用: 18ノード固定なら可能)
        # 任意のつながりを学習できるようにする
        self.bias = nn.Parameter(torch.zeros(1, heads, 18, 18)) 

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
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        head_dim = out_channels // heads
        
        # GraphTransformerLayer に置き換え
        self.conv1 = GraphTransformerLayer(in_channels, head_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GraphTransformerLayer(out_channels, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, adj):
        x, atten1 = self.conv1(x, adj)
        x, _ = self.conv2(x, adj)
        return x, atten1