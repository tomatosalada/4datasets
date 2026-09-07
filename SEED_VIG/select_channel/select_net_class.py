#事前学習用のコード

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.transforms import Compose, Resize, ToTensor
from torch import Tensor
import math
import matplotlib.pyplot as plt
import numpy as np
from GNN.GCN_optunar import DenseGAT3
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj


torch.manual_seed(970530)
torch.cuda.manual_seed_all(970530)



class WeightedAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_heads) / num_heads).cuda()  # 初期値は均等

    def forward(self, attention_weights):
        # ソフトマックスで重み係数を正規化
        normalized_weights = F.softmax(self.weights, dim=0)
        # 各ヘッドのAttention weightに重み係数を掛けて平均
        weighted_avg_attention = sum(
            w * attention.mean(dim=0)
            for w, attention in zip(normalized_weights, attention_weights)
        )
        return weighted_avg_attention



def select_optimal_channels(attention_weights, electrode_attention, num_keep=None, has_cls_token=True):
    """
    重要電極を選定する関数 ([CLS]トークン除外機能付き)

    Args:
        attention_weights: [Batch, Heads, N, N] -> 冗長性の発見に使用
        electrode_attention: [Batch, N] -> 重要度の判定に使用
        num_keep: 最終的に残したい電極数
        has_cls_token (bool): 入力に[CLS]トークンが含まれているか (Default: True)
    
    Returns:
        list: 選定された電極のインデックスリスト (0始まりの電極番号として返します)
    """
    
    # --- 1. データの集約 ---
    # [Batch, Heads, N, N] -> [N, N]
    if attention_weights.dim() == 4:
        avg_attn_matrix = attention_weights.mean(dim=(0, 1)) 
    elif attention_weights.dim() == 3:
        avg_attn_matrix = attention_weights.mean(dim=0)
    else:
        avg_attn_matrix = attention_weights

    # [Batch, N] -> [N]
    if electrode_attention.dim() == 2:
        avg_importance = electrode_attention.mean(dim=0)
    else:
        avg_importance = electrode_attention

    # --- ★追加: [CLS]トークンの除外処理 ---
    if has_cls_token:
        # 行列の [0行目] と [0列目] を削除 -> (18, 18)
        avg_attn_matrix = avg_attn_matrix[1:, 1:]
        # ベクトルの [0番目] を削除 -> (18,)
        avg_importance = avg_importance[1:]
        
        # これにより、インデックス0 は Fp1 (最初の電極) に対応するようになります

    num_electrodes = avg_importance.shape[0]
    
    # --- 2. 冗長なペアの削除 ---
    threshold = avg_attn_matrix.mean() + avg_attn_matrix.std()
    redundant_indices = set()
    
    # 対角成分(自分自身)を0にして無視
    avg_attn_matrix.fill_diagonal_(0)

    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            # ペアの結合強度を確認
            score = (avg_attn_matrix[i, j] + avg_attn_matrix[j, i]) / 2
            
            if score > threshold:
                # 冗長ペア発見：重要度が低い方を削除候補へ
                if avg_importance[i] < avg_importance[j]:
                    redundant_indices.add(i)
                else:
                    redundant_indices.add(j)

    # --- 3. 選定とソート ---
    # 削除リストに含まれないものを抽出
    candidates = [i for i in range(num_electrodes) if i not in redundant_indices]
    
    # 重要度が高い順にソート
    candidates.sort(key=lambda x: avg_importance[x], reverse=True)
    
    # 指定数に絞る
    if num_keep is not None:
        selected_indices = candidates[:num_keep]
    else:
        selected_indices = candidates

    return selected_indices


# depthwise separable convolution(DS Conv):
# depthwise conv + pointwise conv + bn + relu
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernal_size = kernel_size
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)
        #self.bn = nn.BatchNorm2d(ch_out)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        #x = self.bn(x)
        #x = self.relu(x)
        return x

# Context module in DSC module
class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3x3BNReLU, self).__init__()
        self.conv3x3 = depthwise_separable_conv(in_channel, out_channel, 3)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv3x3(x)))

class ContextModule(nn.Module):
    def __init__(self, in_channel):
        super(ContextModule, self).__init__()
        self.stem = Conv3x3BNReLU(in_channel, in_channel // 2)
        self.branch1_conv3x3 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_1 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)
        self.branch2_conv3x3_2 = Conv3x3BNReLU(in_channel // 2, in_channel // 2)

    def forward(self, x):
        x = self.stem(x)
        # branch1
        x1 = self.branch1_conv3x3(x)
        # branch2
        x2 = self.branch2_conv3x3_1(x)
        x2 = self.branch2_conv3x3_2(x2)
        # concat
        return torch.cat([x1, x2], dim=1)

#Transformer
class Config:
    def __init__(self):
        # モデルの設定
        self.hidden_size = 64        # 隠れ層の次元数
        self.num_atention_heads = 16   # アテンションヘッドの数
        self.num_hidden_layers = 4    # Transformer Encoder層の数
        self.intermediate_size = 256  # FeedForwardネットワークの中間層の次元数
        self.hidden_dropout_prob = 0.3 # ドロップアウト率
        self.num_embedding = 17       # SEED-VIGは17チャンネル
        self.num_labels = 2           # SEED-VIGは2クラス

def scaled_dot_product_attention(query, key, value):
    dim_k = torch.tensor(query.size(-1))  # torch.Sizeをテンソルに変換
    scores = torch.bmm(query, key.transpose(1, 2) / torch.sqrt(dim_k))
    weights = F.softmax(scores, dim = -1)
    return torch.bmm(weights, value), weights

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs, weights = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))
        return attn_outputs, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_atention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(num_heads * head_dim, embed_dim)
    
    def forward(self, hidden_state):
        # 各ヘッドからの出力をリストに格納
        attn_outputs = []
        all_weights = []  # 各ヘッドのアテンション重みを格納するリスト

        for h in self.heads:
            attn_output, weights = h(hidden_state)  # 各ヘッドの出力を取得
            attn_outputs.append(attn_output)
            all_weights.append(weights)  # アテンション重みをリストに追加

        x = torch.cat(attn_outputs, dim=2)  # アテンション出力を結合
        x = self.output_linear(x)
        # all_weightsをコピー
        original_all_weights = all_weights.copy()

        # 各ヘッドのAttention weightを平均化
        mean_attention_weights = torch.mean(torch.stack(all_weights), dim=0)  # (batch_size, seq_len, seq_len)

        # 電極ごとのAttention weightを計算
        electrode_attention = torch.mean(mean_attention_weights, dim=1)  # (batch_size, seq_len)


        return x, original_all_weights,  electrode_attention  # 電極ごとのAttention weightを追加

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x):
        #レイヤー正規化を適用し、入力をクエリ、キーバリューにコピー
        hidden_state = self.layer_norm_1(x)
        #スキップ接続付きのアテンションを適用
        # アテンションの出力のみを取得
        x1, attention_weights, electrode_attention = self.attention(hidden_state)  # アテンション重みを取得
        x = x + x1
        # attention_weights をコピー
        original_attention_weights = [weights.clone().cuda() for weights in attention_weights]

        #スキップ接続付きの順伝搬層を適用
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x, original_attention_weights, electrode_attention

class Embeddings(nn.Module):
    def __init__(self, config, gnn_heads=4, gnn_dropout=0.3):
        super().__init__()
        self.config = config
        self.gnn_dropout = gnn_dropout
        self.electrode_embedding = nn.Parameter(torch.randn(1, 17, config.hidden_size))
        
        # カスタムDenseGATを使用
        self.gat3 = DenseGAT3(config.hidden_size, config.hidden_size, heads=gnn_heads, dropout=gnn_dropout)
        
    def forward(self, x, edge_index):
        batch_size = x.size(0)
        device = x.device
        
        adj = to_dense_adj(edge_index, max_num_nodes=17)[0] 
        adj = adj.expand(batch_size, -1, -1).to(device)
        
        # ★修正: タプルで受け取る (x, attention)
        g, gat_attention = self.gat3(x, adj)
        
        x = x + g + self.electrode_embedding

        x = F.dropout(x, p=self.gnn_dropout, training=self.training)
        
        # ★修正: Attentionも返す
        return x, gat_attention

class TransformerEncoder(nn.Module):
    def __init__(self, config, gnn_heads=4, gnn_dropout=0.3):
        super().__init__()
        self.embeddings = Embeddings(config, gnn_heads=gnn_heads, gnn_dropout=gnn_dropout)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # --- 変更点1: クラストークンの定義 ---
        # 形状: [1, 1, hidden_size]
        # バッチサイズに合わせて拡張できるように、最初の次元は1にしておきます
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, x, edge_index, return_attention=False):
        all_attention_weights = []
        all_electrode_attentions = []
        
        # Embeddings (GAT) 通過後の x: [Batch, Num_Electrodes(18), Hidden]
        x, gat_attention = self.embeddings(x, edge_index)
        
        # --- 変更点2: クラストークンを入力の先頭に結合 ---
        batch_size = x.size(0)
        
        # cls_token をバッチサイズ分に拡張: [Batch, 1, Hidden]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # 結合: [Batch, 1 + Num_Electrodes, Hidden] (例: 18 -> 19個のトークンになる)
        x = torch.cat((cls_tokens, x), dim=1)

        for layer in self.layers:
            x, attention_weights, electrode_attention = layer(x)
            
            all_attention_weights.append([weights.clone() for weights in attention_weights])
            all_electrode_attentions.append(electrode_attention)

        # --- 変更点3: 分類用に [CLS] トークンの出力のみを使用 ---
        # x.mean(dim=1) ではなく、先頭(index 0)の特徴量を取得
        cls_output = x[:, 0] 

        if return_attention:
            # 注意: attention_weights や electrode_attention は
            # [CLS]トークンを含んだサイズ (19×19 や 19) になっています
            return cls_output, all_attention_weights, torch.stack(all_electrode_attentions), gat_attention
        else:
            return cls_output
        

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, config):
        super().__init__()
        # global average pooling
        self.clshead = nn.Sequential(
            nn.Linear(emb_size, 32),
            nn.LayerNorm(32),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class spatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)
        spaAtten = q
        spaAtten = torch.squeeze(spaAtten, 1)
        q = self.norm(q)
        # In addition, return to spaAtten for visualization
        return U * q

#shared MLP
class sharedMLP(nn.Module):
    def __init__(self, in_channels):
        super(sharedMLP, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, in_channels // 2)  # 隠れ層の次元は in_channels の半分
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)

    def forward(self, z_avg, z_max):
        # 2つの入力を結合
        z = torch.cat([z_avg, z_max], dim=1)  
        # MLPに適用
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        return z

class frequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.shared = sharedMLP(in_channels)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2,
                                      kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2, in_channels,
                                         kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z_avg = self.avgpool(U)  # (batch_size, in_channels)
        z_max = self.maxpool(U)  # (batch_size, in_channels)
        #z = self.shared(z_avg, z_max)
        z = z_avg + z_max
        z = self.Conv_Squeeze(z)  # 4次元テンソルに戻す
        z = self.Conv_Excitation(z)
        freqAtten = z
        freqAtten = torch.squeeze(freqAtten, 3)
        z = self.norm(z)
        # In addition, return to freqAtten for visualization
        return U * z.expand_as(U)

class sfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.frequencyAttention = frequencyAttention(in_channels)
        self.spatialAttention = spatialAttention(in_channels)
       
    def forward(self, U):
        U_cse = self.frequencyAttention(U)
        U_sse = self.spatialAttention(U)
        
        # Return new 4D features
        # and the Frequency Attention and Spatial_Attention
        return U_cse + U_sse
    
class MultiScalseTemporalConv(nn.Module):
    def __init__(self, in_features, out_channels_per_path, dropout=0.3):
        super().__init__()
        self.dropout_rate = dropout
        self.in_channels = in_features
        self.out_channels = out_channels_per_path

        #---短期カーネル---
        #kernel_size = (時間、周波数)
        self.conv_short = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 1), #(time, freq)
            padding=(1, 0), #(pad_time, pad_freq)
            dilation=(1, 1)
        )

        #---中期カーネル---
        # RF: 5 (Dilated)
        self.conv_mid = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 1),
            padding=(2, 0),
            dilation=(2, 1)
        )

        #---長期カーネル---
        # RF: 5 (Dense)
        self.conv_long = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(5, 1),
            padding=(2, 0),
            dilation=(1, 1)
        )

        #---全体カーネル---
        # RF: 7 (Covers entire seq_len=6)
        self.conv_all = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(7, 1),
            padding=(3, 0),
            dilation=(1, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels_per_path * 4)
        total_channels = out_channels_per_path * 4
        self.fusion_conv = nn.Conv2d(out_channels_per_path * 4, out_channels_per_path * 4, kernel_size=1)
    
    def forward(self, x):
        #(batch, node_dim, time, freq) = (100, 18, 16, 5)
        out_short = F.relu(self.conv_short(x))
        out_short = F.dropout(out_short, p=self.dropout_rate, training=self.training)
        #print(out_short.shape)
        out_mid = F.relu(self.conv_mid(x))
        out_mid = F.dropout(out_mid, p=self.dropout_rate, training=self.training)
        #print(out_mid.shape)
        out_long = F.relu(self.conv_long(x))
        out_long = F.dropout(out_long, p=self.dropout_rate, training=self.training)
        #print(out_long.shape)
        out_all = F.relu(self.conv_all(x))
        out_all = F.dropout(out_all, p=self.dropout_rate, training=self.training)
        #print(out_all.shape)

        output = torch.cat([out_short, out_mid, out_long, out_all], dim=1)

        out = self.bn(output)

        out = F.relu(self.fusion_conv(out))

        return out

# Toshi-Net:
# Attention module + DSC module + transformer module
class gcn_select_net(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64, num_hidden_layers=4, 
                 transformer_dropout=0.3, cnn_dropout=0.3, gnn_dropout=0.3, 
                 num_attention_heads=16, gnn_heads=4, cnn_out_channels=16):
        super(gcn_select_net, self).__init__()
        self.cnn_dropout = cnn_dropout

        self.conv_time = MultiScalseTemporalConv(1, cnn_out_channels, dropout=cnn_dropout)
        cnn_hidden = cnn_out_channels * 4

        self.Atten = sfAttention(in_channels=cnn_hidden) 

        self.conv2d = nn.Conv2d(in_channels=cnn_hidden, out_channels=cnn_hidden//2, kernel_size=(3, 3), padding=(1, 1))
        self.bm2d = nn.BatchNorm2d(cnn_hidden//2, affine=False)
        self.global_pool = nn.AdaptiveMaxPool2d((4, 2))
        self.conv2d2 = nn.Conv2d(in_channels=cnn_hidden//2, out_channels=cnn_hidden//4, kernel_size=(3, 3), padding=(1, 1))
        self.bm2d2 = nn.BatchNorm2d(cnn_hidden//4, affine=False)
        self.global_pool2 = nn.AdaptiveAvgPool2d((2, 2))

        # SEED-VIG用のグラフ
        self.graph_path = "/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/GNN/graph_path/common_graph_edge_index_full_17ch.pt"
        self.edge_index = torch.load(self.graph_path, weights_only=False)

        #transformer
        config = Config()
        config.hidden_size = hidden_size
        config.num_hidden_layers = num_hidden_layers
        config.hidden_dropout_prob = transformer_dropout
        config.num_atention_heads = num_attention_heads
        config.intermediate_size = hidden_size * 4
        
        cnn_flatten_dim = (cnn_hidden // 4) * 4
        self.transformer_proj = nn.Linear(cnn_flatten_dim, hidden_size) if cnn_flatten_dim != hidden_size else nn.Identity()
        
        self.transformer = TransformerEncoder(config, gnn_heads=gnn_heads, gnn_dropout=gnn_dropout)
        self.classification = ClassificationHead(config.hidden_size, num_classes, config)


    def forward(self, x):
        #[batch、時系列、周波数、特徴量（電極）]→x.shape torch.Size([10, 60, 5, 32])
        batch_size, seq_len, freq_len, node_dim = x.shape  # 各次元のサイズを取得
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(-1, seq_len, freq_len) #[320, 60, 5]
        x = x.unsqueeze(1) #[320, 1, 60, 5]
        x_time = self.conv_time(x) #[320, 96, 60, 5]
        #x_time = x_time.view(batch_size, node_dim, -1, seq_len, freq_len) #torch.Size([100, 18, 32, 16, 5])
        #print(x_time.shape)

        x_atten = self.Atten(x_time)   
        
        x_atten = x_atten + x_time

        #print(x_atten.shape)
        x_atten = x_atten.view(batch_size * node_dim, -1, seq_len, freq_len)  

        x_atten = self.conv2d(x_atten)
        x_atten = self.bm2d(x_atten)
        x_atten = F.relu(x_atten)
        x_atten = F.dropout(x_atten, p=self.cnn_dropout, training=self.training)
        

        x_atten = self.conv2d2(x_atten)
        x_atten = self.bm2d2(x_atten)
        x_atten = F.relu(x_atten)
        x_atten = self.global_pool2(x_atten)
        x_atten = F.dropout(x_atten, p=self.cnn_dropout, training=self.training)
        
        #print(x_atten.shape)

        x_atten = x_atten.view(batch_size, node_dim, -1)
        x_atten = self.transformer_proj(x_atten)

        #Transformer
        # 出力とアテンション重みの取得
        # edge_indexをxと同じデバイスに移動
        edge_index = self.edge_index.to(x.device)
        out, attention_weights, electrode_attention, gat_attention = self.transformer(x_atten, edge_index, return_attention=True)  # return_attention=True を追加
        #out -> [batch, 64]
        out = self.classification(out)

        return out, attention_weights, electrode_attention, gat_attention 

if __name__ == '__main__':
    # SEED-VIG ダミーデータ: (Batch, Time, Freq, Channel) = (100, 16, 5, 17)
    input = torch.rand((100, 16, 5, 17))
    net = gcn_select_net(num_classes=3)
    
    output, attention_weights, electrode_attention, gat_attention = net(input)
    print("Input shape     : ", input.shape)
    print("Output shape    : ", output.shape)
    print("Atten shape  : ", electrode_attention.shape)
    # 冗長な電極を削除
    removed_electrodes = select_optimal_channels(attention_weights, electrode_attention)
    print("Removed electrodes:", removed_electrodes)