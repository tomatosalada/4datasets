"""
TUAB (2クラス) 固定電極/動的電極の割合スイープ。

基本の処理 (データ読み込み・holdout学習ループ・Val Accuracy 最良エポックでのモデル選定・
可視化・CSV集約・並列実行) は従来どおり。
【動的な電極選択 (Gate) だけ】を
research_number_of_electrode/test_eegnet_researchbestnum_forpaper_eeggate.py と
同じ設計に差し替えてある。詳細は下の「★ Gate (動的な電極選択) の設計」を参照。
"""

# --- ライブラリのインポート ---
import sys
import os
# プロジェクトルートを import パスに追加 (select_channel / model を解決するため)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import argparse
import traceback
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 並列プロセス / ヘッドレス環境で描画するため非GUIバックエンドを使用
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score,
                             cohen_kappa_score, balanced_accuracy_score)
import warnings
from matplotlib.patches import Ellipse
from scipy.interpolate import griddata
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

# --- 自作モジュールのインポート ---
# cheat_sheet (gcn_select_net) は optuna trial 35 の構造で学習した TUAB の
# gcn_select_net_holdout.pth をロードするため、trial 35 のパラメータを
# デフォルトに反映済みの select_net_class_optunar を使用する。
from select_channel.select_net_class_optunar import *
from model.EEGNet import CustomEEGNet

warnings.filterwarnings("ignore")

# =========================================================================
# ★ 「固定電極の割合」スイープ設定 (モジュール全体で共有する)
#
#   1. 総電極数 (TOTAL_ELECTRODES) を指定する。
#   2. 事前学習で得られた電極の重要度ランキング (IMPORTANCE_RANKING を参照。
#      holdout_eval_results.txt の "Electrode Importance Ranking" をコードに直接
#      埋め込んだもの) の上位から順に「固定する電極」を選ぶ。
#   3. 固定する電極数を 0〜総電極数まで 1 ずつ増やしながら学習・評価する。
#      - 固定数 F のとき: 上位 F 電極を固定 (常時ON)、残り (TOTAL-F) 電極を
#        Gate で動的選定する (可変チャネルから TOTAL-F 個を選ぶ)。
#      - F=0        : 全 TOTAL 電極を動的選定 (固定なし)
#      - F=TOTAL    : 上位 TOTAL 電極を固定 (動的選定なし)
#   4. 各結果を CSV に集約し、比較しやすい図を作成する (DBは作らない)。
#
#   ※ ハイパーパラメータ (batch_size / LR / weight_decay / lambda_sparsity /
#      使用モデル model.EEGNet) と「Val Accuracy 最良エポックでのモデル選定」は
#      元コードから変更していない。
# =========================================================================
SEED = 42
BATCH_SIZE = 32
DEFAULT_EPOCH = 100
DEFAULT_TOTAL_ELECTRODES = 11  # コマンドライン --total で上書き可能

# TUAB の cheat sheet / classifier 学習に用いたハイパーパラメータ
# (参考: research_number_of_electrode/test_eegnet_researchbestnum_forpaper.py)
LR_GATE = 0.00045885300055577613
LR_CLASSIFIER = 0.00045885300055577613
WEIGHT_DECAY = 0.0021530469806865446
LAMBDA_SPARSITY = 1e-3

PRETRAINED_MODEL_PATH = '/mnt/data/toshiki.ohno/TUAB/test_classification/result_optuna_holdout/gcn_select_net_holdout.pth'

# 探索結果一式の出力先ルート (総電極数ごとに results_channel{N}/ を作る)
RESEARCH_ROOT_DIR = '/mnt/data/toshiki.ohno/TUAB/research_ratio_fixdynamic/eeggate/'

# CHANNEL_NAMES は「データのチャネル次元(=16)の物理的な並び順」。
# 重要度ランキングはこの名前で電極を指すため、名前→インデックス変換に使う。
CHANNEL_NAMES = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
]
NUM_CHANNELS = len(CHANNEL_NAMES)

# 固定電極の選定順 (重要度が高い順)。
# 事前学習の成果物 (TUAB) holdout_eval_results.txt の
# "Electrode Importance Ranking" をそのままコードに埋め込んだもの。
# 上位から順に「固定する電極」に採用する。
IMPORTANCE_RANKING = [
    'T6-O2',   #  1: 0.068489
    'FP2-F4',  #  2: 0.067681
    'P4-O2',   #  3: 0.063406
    'P3-O1',   #  4: 0.062547
    'T4-T6',   #  5: 0.061485
    'FP1-F7',  #  6: 0.061369
    'FP1-F3',  #  7: 0.060780
    'T3-T5',   #  8: 0.059500
    'FP2-F8',  #  9: 0.058494
    'F8-T4',   # 10: 0.058116
    'F7-T3',   # 11: 0.058108
    'T5-O1',   # 12: 0.057227
    'F3-C3',   # 13: 0.054775
    'F4-C4',   # 14: 0.051155
    'C4-P4',   # 15: 0.051109
    'C3-P3',   # 16: 0.047793
]
# ランキングが 16ch を過不足なく網羅していることを保証する
assert sorted(IMPORTANCE_RANKING) == sorted(CHANNEL_NAMES), \
    "IMPORTANCE_RANKING が CHANNEL_NAMES と一致しません (電極名の重複/欠落を確認)"
# TUAB は正常/異常の 2 クラス (ラベル 0/1)
CLASS_LABELS = {0: 'Normal', 1: 'Abnormal'}

# 1つのGPUに載せる並列プロセス数。モデルは軽量なため 2 で安全側。
# GPUメモリに余裕があれば更に増やして高速化できる。
PROCESSES_PER_GPU = 6

# 入力の周波数帯数 (データ形状 [N, 20(time), 5(freq), 16(channel)] の freq 軸)
NUM_BANDS = 5

# =========================================================================
# ★ Gate (動的な電極選択) の設計
#   research_number_of_electrode/test_eegnet_researchbestnum_forpaper_eeggate.py
#   (元は loso_researchbestnum_paper2class_eeggate.py) の改修をそのまま移植したもの。
#   ここ以外の処理は移植前の ratio_fixdynamic_forpaper_eeggate.py と同一。
#
#   移植前の実装では、分類器の出力がゲートの二値マスクだけの関数になっており、その
#   マスクは per-sample の teacher_importance が決めていて EEG はほとんど寄与して
#   いなかった (SEED-VIG 実測: x を定数化してもマスクのビット一致 99.1% /
#   maskonly 予測一致 95.3%)。原因ごとに値を変えてある。
#   旧挙動は GATE_FEAT='mean' / GATE_TEACHER='per_sample' / GATE_HINTS=True /
#   SELECT='threshold' / MASK_RANDOMIZE=0.0 で再現できる。
# =========================================================================
GATE_FEAT = 'band'           # Gate に渡す EEG 特徴。band: 時間だけ平均し帯域を残す
                             # [B,20,5,16] -> [B,5,16] -> 80次元
                             # mean(旧): 時間も帯域も平均して [B,16]。帯域差が消える
GATE_TEACHER = 'train_const'  # 教師の電極重要度の入れ方。
                             # train_const: 学習データだけの平均を1本作り全サンプルに配る
                             #              -> per-sample の通信路が消え、動くのは EEG だけ
                             #              (LOSO 版の fold_const に対応。holdout なので
                             #               fold 平均ではなく train split の平均)
                             # per_sample(旧): サンプルごとの重要度をそのまま渡す
                             # none: 教師を Gate 入力から外す
GATE_HINTS = False           # 教師の予測確率を Gate に入れるか。True(旧)は答えを直接渡すことになる
SELECT = 'topk'              # topk: スコア上位k本を厳密に選ぶ (straight-through)
                             # threshold(旧): sigmoid>0.5 + 疎性罰則。目標本数に届かない
MASK_RANDOMIZE = 0.3         # 学習時にこの確率でマスクをランダムなk本に差し替える。
                             # マスクと正解の相関を壊し、分類器がマスクを符号として
                             # 読むのを防ぐ。0で無効(旧)


# =========================================================================
# 0. Visualization Helpers
# =========================================================================
def get_electrode_coords():
    # TUEV 16 Channels coordinates
    coords = {
        'FP1-F7': (-0.6, 0.7), 'F7-T3': (-0.8, 0.3), 'T3-T5': (-0.8, -0.3), 'T5-O1': (-0.6, -0.7),
        'FP2-F8': (0.6, 0.7), 'F8-T4': (0.8, 0.3), 'T4-T6': (0.8, -0.3), 'T6-O2': (0.6, -0.7),
        'FP1-F3': (-0.3, 0.7), 'F3-C3': (-0.3, 0.0), 'C3-P3': (-0.3, -0.5), 'P3-O1': (-0.3, -0.7),
        'FP2-F4': (0.3, 0.7), 'F4-C4': (0.3, 0.0), 'C4-P4': (0.3, -0.5), 'P4-O2': (0.3, -0.7)
    }
    return coords

def plot_head_map(ax, channel_names, usage_weights, vmin, vmax, title=""):
    coords = get_electrode_coords()
    x_coords = []
    y_coords = []
    z_values = []

    for i, name in enumerate(channel_names):
        if name in coords:
            x, y = coords[name]
            x_coords.append(x)
            y_coords.append(y)
            z_values.append(usage_weights[i])

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_values = np.array(z_values)

    angles = np.linspace(0, 2*np.pi, 36)
    edge_x = np.cos(angles)
    edge_y = np.sin(angles)
    edge_z = []

    for ex, ey in zip(edge_x, edge_y):
        dists = np.sqrt((x_coords - ex)**2 + (y_coords - ey)**2)
        nearest_idx = np.argmin(dists)
        edge_z.append(z_values[nearest_idx])

    x_coords = np.concatenate([x_coords, edge_x])
    y_coords = np.concatenate([y_coords, edge_y])
    z_values = np.concatenate([z_values, np.array(edge_z)])

    grid_x, grid_y = np.mgrid[-1.2:1.2:300j, -1.2:1.2:300j]
    grid_z = griddata((x_coords, y_coords), z_values, (grid_x, grid_y), method='cubic', fill_value=vmin)

    grid_z = np.clip(grid_z, 0.0, 1.0)

    dist = np.sqrt(grid_x**2 + grid_y**2)
    radius = 1.0
    grid_z = np.ma.masked_where(dist > radius, grid_z)

    levels = np.linspace(0.0, 1.0, 100)
    im = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='jet', vmin=0.0, vmax=1.0)

    circle = plt.Circle((0, 0), 1.0, color='k', fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k', linewidth=2)

    for i, name in enumerate(channel_names):
        if name in coords:
            cx, cy = coords[name]
            ax.scatter(cx, cy, c='k', s=20, marker='o', alpha=0.5)
            ax.text(cx, cy, name, fontsize=8, ha='center', va='center', color='black', alpha=0.7)

    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.4)
    ax.set_aspect('equal')
    return im

def load_data():
    # ★データ変更: TUEV(6クラス) -> TUAB(2クラス)。
    #   test_classification/test_optunar_holdout.py の load_data と同じ方式に合わせる。
    #   ・TUAB_data 直下の {train,val,test}_{data,labels}.npy を直接ロード
    #   ・ラベルは 0/1 のためそのまま (TUEV のような "-1" は行わない)
    #   ・形状 [N, 16(channel), 20(time), 5(freq)] -> [N, 20(time), 5(freq), 16(channel)]
    data_dir = os.environ.get(
        "TUAB_DATA_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TUAB_data"))
    )

    X_train_np = np.load(os.path.join(data_dir, "train_data.npy"))
    y_train_np = np.load(os.path.join(data_dir, "train_labels.npy")).squeeze().astype(int)

    X_val_np = np.load(os.path.join(data_dir, "val_data.npy"))
    y_val_np = np.load(os.path.join(data_dir, "val_labels.npy")).squeeze().astype(int)

    X_eval_np = np.load(os.path.join(data_dir, "test_data.npy"))
    y_eval_np = np.load(os.path.join(data_dir, "test_labels.npy")).squeeze().astype(int)

    num_classes = len(np.unique(y_train_np))

    # [N, 16(channel), 20(time), 5(freq)] -> [N, 20(time), 5(freq), 16(channel)]
    X_train_np = np.transpose(X_train_np, (0, 2, 3, 1))
    X_val_np = np.transpose(X_val_np, (0, 2, 3, 1))
    X_eval_np = np.transpose(X_eval_np, (0, 2, 3, 1))

    # Data Leakageを防ぐため、Trainデータのみで正規化を計算
    mean = np.mean(X_train_np, axis=0, keepdims=True)
    std = np.std(X_train_np, axis=0, keepdims=True)

    X_train_np = (X_train_np - mean) / (std + 1e-8)
    X_val_np = (X_val_np - mean) / (std + 1e-8)
    X_eval_np = (X_eval_np - mean) / (std + 1e-8)

    X_train = torch.FloatTensor(X_train_np)
    y_train = torch.LongTensor(y_train_np)
    X_val = torch.FloatTensor(X_val_np)
    y_val = torch.LongTensor(y_val_np)
    X_eval = torch.FloatTensor(X_eval_np)
    y_eval = torch.LongTensor(y_eval_np)

    return X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes

# =========================================================================
# 1. Gate Mechanism
#   ★ test_eegnet_researchbestnum_forpaper_eeggate.py から移植
# =========================================================================
class GateMechanism(nn.Module):
    """受信した EEG (+ 教師の事前分布) から、どの電極を使うかのスコアを出す全結合層。

    入力の内訳は呼び出し側が決める:
      feat        [B, feat_dim]  EEG 特徴。GATE_FEAT='band' なら帯域を残した 5x16=80次元
      hints       [B, C]         教師の予測確率。GATE_HINTS=False なら渡さない
      teacher_attn[B, 16]        教師の電極重要度。GATE_TEACHER='train_const' なら
                                 全サンプル同じ定数なので per-sample の情報は持たない
    """

    def __init__(self, feat_dim, num_variable_channels, hint_dim=0, imp_dim=0):
        super(GateMechanism, self).__init__()
        input_dim = feat_dim + hint_dim + imp_dim
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_variable_channels),
            nn.Sigmoid()
        )

    def forward(self, feat, hints=None, teacher_attn=None):
        parts = [feat]
        if hints is not None:
            parts.append(hints)
        if teacher_attn is not None:
            parts.append(teacher_attn)
        return self.gate_net(torch.cat(parts, dim=1))

# =========================================================================
# 2. ConceptDynamicNet
#   ★ test_eegnet_researchbestnum_forpaper_eeggate.py から移植
# =========================================================================
@torch.no_grad()
def teacher_outputs(cheat_sheet_model, x):
    """教師モデルから Gate に渡す2つの量を計算する (移植前と同じ計算)。

    returns: teacher_hints [B, C] (softmax確率), teacher_importance [B, 16] (電極方向のz-score)
    """
    teacher_logits, _, electrode_attention, _ = cheat_sheet_model(x)
    teacher_hints = torch.softmax(teacher_logits, dim=1)

    # electrode_attention shape: [Num_Layers, Batch, 17]
    avg_layer_attn = electrode_attention.mean(dim=0)
    teacher_importance = avg_layer_attn[:, 1:]   # [CLS]トークン除外

    mean_imp = teacher_importance.mean(dim=1, keepdim=True)
    std_imp = teacher_importance.std(dim=1, keepdim=True)
    teacher_importance = (teacher_importance - mean_imp) / (std_imp + 1e-6)
    return teacher_hints, teacher_importance


@torch.no_grad()
def train_constant_importance(cheat_sheet_model, dataloader, num_channels, device):
    """Gate に渡す教師の電極重要度 [16] を、【学習データだけ】から作る。

    per-sample の重要度をそのまま Gate に渡すと、それがサンプルごとの通信路になって
    マスクの中身を決めてしまう (SEED-VIG 実測: imp を定数化するとマスクの種類が
    36 -> 7 に潰れる。逆に EEG を定数化してもビット一致 99.1% で何も変わらない)。
    学習データ全体の定数にすれば「どの電極が有用か」という事前分布だけが残り、
    サンプルごとに動くのは EEG だけになる。val / eval のデータは見ない。
    """
    cheat_sheet_model.eval()
    total = torch.zeros(num_channels, device=device)
    n_seen = 0
    for x_batch, _ in dataloader:
        x_batch = x_batch.to(device)
        _, imp = teacher_outputs(cheat_sheet_model, x_batch)
        total += imp.sum(dim=0)
        n_seen += x_batch.size(0)
    return total / max(n_seen, 1)


class ConceptDynamicNet(nn.Module):
    """受信した EEG からその都度electrodeを選び、固定電極と合わせて EEGNet に分類させる。

    経路は「EEG (+ 学習データ定数の教師事前分布) -> 全結合層 -> 上位k本を選択 -> EEGNet」。
    固定電極 (fixed_indices) は常時 ON で、可変電極 (variable_indices) の中から
    n_select_var 本を Gate が選ぶ。

    【マスクを符号として使わせないための仕掛け】
    EEGNet の conv2 は kernel=(channel*freq, 1)、つまり 電極x帯域 の80次元を一発で
    線形結合する。ゼロ埋めマスクはこの線形和に「どの行が0か」を直接見せるので、x が
    定数でも出力をマスクから復号できてしまう (移植前はこれが起きていた。SEED-VIG 実測
    では x を定数に潰しても予測が 95.3%一致した)。
    そこで学習時に確率 mask_randomize でマスクをランダムなk本に差し替え、マスクと正解の
    相関を壊す。差し替えた分は Gate に勾配が流れない (定数扱い) ので、分類器だけが
    「マスクの模様は当てにならない」と学ぶ。

    top-k はこの仕掛けの前提でもある。選択本数が揺れると、分類器が「立っている本数」
    だけで学習マスクとランダムマスクを見分けられてしまい、正則化が効かなくなる。
    """

    def __init__(self, cheat_sheet_model, classifier_model, num_channels=16,
                 fixed_indices=[], variable_indices=[], num_classes=6,
                 n_select_var=None, gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER,
                 use_hints=GATE_HINTS, select=SELECT, mask_randomize=MASK_RANDOMIZE,
                 n_bands=NUM_BANDS):
        super(ConceptDynamicNet, self).__init__()

        self.cheat_sheet = cheat_sheet_model
        for param in self.cheat_sheet.parameters():
            param.requires_grad = False

        self.fixed_indices = fixed_indices
        self.variable_indices = variable_indices
        self.num_channels = num_channels
        self.gate_feat = gate_feat
        self.gate_teacher = gate_teacher
        self.use_hints = use_hints
        self.select = select
        self.mask_randomize = mask_randomize
        # 選ぶ可変チャネル数。None なら top-k を使わず閾値方式にフォールバックする
        # (固定数=総電極数のときは 0。可変電極を1本も選ばない)
        self.n_select_var = n_select_var

        # train_const 用の定数 [16]。set_teacher_const() で学習データから入れる。
        self.register_buffer('teacher_const', torch.zeros(num_channels))

        imp_dim = 0 if gate_teacher == 'none' else num_channels
        feat_dim = num_channels * n_bands if gate_feat == 'band' else num_channels
        self.gate = GateMechanism(feat_dim=feat_dim,
                                  num_variable_channels=len(variable_indices),
                                  hint_dim=num_classes if use_hints else 0,
                                  imp_dim=imp_dim)
        self.classifier = classifier_model

    def set_teacher_const(self, imp):
        """train_constant_importance() が作った [16] を保持する。"""
        self.teacher_const.copy_(imp.to(self.teacher_const.device))

    def gate_features(self, x):
        """Gate に渡す EEG 特徴。x: [B, 20(time), 5(freq), 16(channel)]"""
        if self.gate_feat == 'band':
            # 時間だけ平均し、帯域は残す -> [B, 5, 16] -> [B, 80]
            return x.mean(dim=1).reshape(x.size(0), -1)
        # 旧挙動: 時間も帯域も平均 -> [B, 16]
        return x.mean(dim=(1, 2))

    def _select(self, scores):
        """スコアから二値マスクを作る。forward の値は厳密に二値で、勾配は
        straight-through でスコアに流す。"""
        k = self.n_select_var
        n_var = scores.size(1)
        if self.select == 'topk' and k is not None:
            if k <= 0:
                # 固定電極だけで総電極数に達している場合 (num_fixed == total)
                hard = torch.zeros_like(scores)
            elif k >= n_var:
                hard = torch.ones_like(scores)
            else:
                idx = scores.topk(k, dim=1).indices
                hard = torch.zeros_like(scores).scatter(1, idx, 1.0)
        else:
            hard = (scores > 0.5).float()
        return hard - scores.detach() + scores

    def _random_mask(self, like):
        """各サンプルについて、学習マスクと同じ本数のランダムな部分集合を作る。"""
        k = self.n_select_var
        n_var = like.size(1)
        if k is None or k >= n_var:
            return torch.ones_like(like)
        if k <= 0:
            return torch.zeros_like(like)
        idx = torch.rand_like(like).topk(k, dim=1).indices
        return torch.zeros_like(like).scatter(1, idx, 1.0)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # 教師を実際に引く必要があるのは、予測確率を使うときと per_sample のときだけ。
        # train_const / none では教師出力はマスクに影響しないので呼ばない。
        teacher_hints, teacher_importance = None, None
        if self.use_hints or self.gate_teacher == 'per_sample':
            with torch.no_grad():
                teacher_hints, teacher_importance = teacher_outputs(self.cheat_sheet, x)

        if self.gate_teacher == 'train_const':
            # 学習データ平均の1本を、バッチ内の全サンプルに配る (per-sample の通信路なし)
            teacher_attn = self.teacher_const.unsqueeze(0).expand(batch_size, -1)
            teacher_importance = teacher_attn
        elif self.gate_teacher == 'per_sample':
            teacher_attn = teacher_importance
        else:
            teacher_attn = None

        feat = self.gate_features(x)
        gate_weights_soft_var = self.gate(feat,
                                          teacher_hints if self.use_hints else None,
                                          teacher_attn)
        gate_weights_var = self._select(gate_weights_soft_var)

        # マスクの結合処理 (Fixed=1.0 + Variable=Gate)
        full_gate_weights = torch.ones(batch_size, self.num_channels).to(device)
        if len(self.variable_indices) > 0:
            full_gate_weights[:, self.variable_indices] = gate_weights_var

        # 分類器に渡すマスク。学習時のみ一部をランダムなk本に差し替える。
        # 集計・可視化に返すのは差し替え前の「学習したマスク」のほう。
        applied = full_gate_weights
        if self.training and self.mask_randomize > 0 and len(self.variable_indices) > 0:
            swap = torch.rand(batch_size, device=device) < self.mask_randomize
            if bool(swap.any()):
                rnd_var = self._random_mask(gate_weights_var.detach())
                rnd_full = torch.ones(batch_size, self.num_channels, device=device)
                rnd_full[:, self.variable_indices] = rnd_var
                applied = torch.where(swap.unsqueeze(1), rnd_full, full_gate_weights)

        # マスク適用
        weights_expanded = applied.unsqueeze(1).unsqueeze(1)
        x_masked = x * weights_expanded

        outputs = self.classifier(x_masked)

        return outputs, gate_weights_var, full_gate_weights, teacher_importance

# =========================================================================
# 学習ループ (Holdout用)
#   ※ モデル選定は元コード通り「Val Accuracy 最良エポック」。
#      Balanced Accuracy は比較のため Eval で追加計算するのみ。
# =========================================================================
def train_eval_holdout(X_train, y_train, X_val, y_val, X_eval, y_eval,
                       num_classes, num_channels, fixed_indices, variable_indices,
                       batch_size, epoch, target_var_electrodes, run_label,
                       lambda_sparsity, lr_gate, lr_classifier,
                       pretrained_model_path, save_path, device,
                       save_model=True, verbose=True):

    if verbose:
        print(f"\n======================================", flush=True)
        print(f"   [{run_label}] Starting Holdout Training on {device}", flush=True)
        print(f"======================================", flush=True)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    eval_dataset = TensorDataset(X_eval, y_eval)

    y_train_list = y_train.tolist()
    counts = Counter(y_train_list)
    total = len(y_train_list)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    cheat_sheet_model = gcn_select_net(num_classes=num_classes).to(device)

    if os.path.exists(pretrained_model_path):
        if verbose:
            print(f"Loading Cheat Sheet (Pre-trained): {pretrained_model_path}", flush=True)
        cheat_sheet_model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    else:
        if verbose:
            print(f"【注意】事前学習モデルが見つかりません: {pretrained_model_path}", flush=True)

    # データ形状 [N, seq_len(time), freq, channel] から入力サイズを動的決定
    # (TUEV: time=10 / TUAB: time=20 と異なるため固定値にしない)
    seq_len = X_train.shape[1]
    freq = X_train.shape[2]
    classifier_model = CustomEEGNet(numclasses=num_classes, seq_len=seq_len, freq=freq, channel=num_channels).to(device)

    model = ConceptDynamicNet(
        cheat_sheet_model,
        classifier_model,
        num_channels=num_channels,
        fixed_indices=fixed_indices,
        variable_indices=variable_indices,
        num_classes=num_classes,
        n_select_var=target_var_electrodes,   # ★ top-k で厳密にこの本数だけ選ぶ
        gate_feat=GATE_FEAT,
        gate_teacher=GATE_TEACHER,
        use_hints=GATE_HINTS,
        select=SELECT,
        mask_randomize=MASK_RANDOMIZE,
        n_bands=freq,
    ).to(device)

    print(f"[{run_label}] gate_feat={GATE_FEAT} | "
          f"gate_teacher={GATE_TEACHER} | gate_hints={GATE_HINTS} | "
          f"select={SELECT} | mask_rand={MASK_RANDOMIZE}", flush=True)

    # ★ 教師の電極重要度は【学習データだけ】から作った定数を全サンプルに配る。
    #   val / eval は見ないのでリークしない。1回だけ計算して以降のエポックで使い回す。
    if GATE_TEACHER == 'train_const':
        imp_const = train_constant_importance(cheat_sheet_model, train_dataloader,
                                              num_channels, device)
        model.set_teacher_const(imp_const)
        top = torch.argsort(imp_const, descending=True)[:5].tolist()
        print(f"[{run_label}] teacher importance (train mean) "
              f"top5: {[CHANNEL_NAMES[i] for i in top]}", flush=True)

    optimizer = torch.optim.AdamW([
        {'params': model.gate.parameters(), 'lr': lr_gate},
        {'params': model.classifier.parameters(), 'lr': lr_classifier}
    ], weight_decay=WEIGHT_DECAY)

    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn_main = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_val_elec = 0.0
    best_model_state = None

    best_class_gate_sum = torch.zeros(num_classes, num_channels).to(device)
    best_class_counts = torch.zeros(num_classes).to(device)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for i in range(epoch):
        # --- Train Loop ---
        model.train()
        model.cheat_sheet.eval()
        total_train_loss = 0
        total_train_correct = 0

        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs, gate_weights_var, full_gate_weights, _ = model(x_batch)
            loss_cls = loss_fn_main(outputs, y_batch)

            if SELECT == 'topk':
                # ★ top-k は常に厳密に k 本なので疎性罰則は不要 (罰則ゼロと同じ)
                loss = loss_cls
            else:
                usage_per_sample_var = gate_weights_var.sum(dim=1)
                loss_sparsity = ((usage_per_sample_var - target_var_electrodes) ** 2).mean()
                loss = loss_cls + (lambda_sparsity * loss_sparsity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_train_correct += (predicted == y_batch).sum().item()

        epoch_train_loss = total_train_loss / len(train_dataloader)
        epoch_train_acc = 100.0 * total_train_correct / len(train_dataset)

        # --- Val Loop ---
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        val_full_gate_sum = 0

        temp_class_gate_sum = torch.zeros(num_classes, num_channels).to(device)
        temp_class_counts = torch.zeros(num_classes).to(device)

        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                outputs, gate_weights_var, full_gate_weights, _ = model(x_batch)
                loss = loss_fn_main(outputs, y_batch)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val_correct += (predicted == y_batch).sum().item()

                val_full_gate_sum += full_gate_weights.sum(dim=1).mean().item()

                for c in range(num_classes):
                    mask = (y_batch == c)
                    if mask.sum() > 0:
                        selected_gates = full_gate_weights[mask]
                        temp_class_gate_sum[c] += selected_gates.sum(dim=0)
                        temp_class_counts[c] += mask.sum()

        epoch_val_loss = total_val_loss / len(val_dataloader)
        epoch_val_acc = 100.0 * total_val_correct / len(val_dataset)
        avg_val_elec = val_full_gate_sum / len(val_dataloader)

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)
        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)

        if verbose and (i + 1) % 1 == 0:
            print(f"[{run_label}] Epoch {i+1:03d}/{epoch} | "
                  f"Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}% | "
                  f"Usage: {avg_val_elec:.1f}ch", flush=True)

        # ★ 元コード通り Val Accuracy が最良のエポックのモデルを保存対象にする
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_elec = avg_val_elec
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_class_gate_sum = temp_class_gate_sum.clone()
            best_class_counts = temp_class_counts.clone()

    if verbose:
        print(f"[{run_label}] Best Val Accuracy: {best_val_acc:.2f}% "
              f"(Usage: {best_val_elec:.1f}ch)", flush=True)
    if save_model and best_model_state is not None:
        torch.save(best_model_state, f"{save_path}concept_dynamic_net_holdout.pth")

    # ----------------------------------------------------
    # FINAL TEST ON EVAL SET
    # ----------------------------------------------------
    if verbose:
        print(f"[{run_label}] Testing best model on EVAL set...", flush=True)
    model.load_state_dict(best_model_state)
    model.eval()

    eval_true = []
    eval_pred = []

    with torch.no_grad():
        for x_batch, y_batch in eval_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs, _, _, _ = model(x_batch)
            _, preds = torch.max(outputs, 1)
            eval_true.extend(y_batch.cpu().tolist())
            eval_pred.extend(preds.cpu().tolist())

    eval_acc = accuracy_score(eval_true, eval_pred) * 100.0
    eval_bacc = balanced_accuracy_score(eval_true, eval_pred)
    eval_prec = precision_score(eval_true, eval_pred, average='macro', zero_division=0)
    eval_rec = recall_score(eval_true, eval_pred, average='macro', zero_division=0)
    eval_f1 = f1_score(eval_true, eval_pred, average='macro', zero_division=0)
    eval_kap = cohen_kappa_score(eval_true, eval_pred)

    if verbose:
        print(f"[{run_label}] Eval Results -> "
              f"Acc: {eval_acc:.2f}%, BACC: {eval_bacc:.4f}, F1: {eval_f1:.4f}", flush=True)

    result = {
        'val_acc': best_val_acc,
        'val_elec': best_val_elec,
        'eval_acc': eval_acc,
        'eval_bacc': eval_bacc,
        'eval_prec': eval_prec,
        'eval_rec': eval_rec,
        'eval_f1': eval_f1,
        'eval_kap': eval_kap,
        'eval_true': eval_true,
        'eval_pred': eval_pred,
        'train_loss_hist': train_loss_history,
        'val_loss_hist': val_loss_history,
        'train_acc_hist': train_acc_history,
        'val_acc_hist': val_acc_history,
        'class_gate_sum': best_class_gate_sum.cpu(),
        'class_counts': best_class_counts.cpu()
    }

    del model
    del cheat_sheet_model
    del optimizer
    torch.cuda.empty_cache()

    return result

# =========================================================================
# 固定電極数ごとの成果物 (プロット・テキスト) 保存
# =========================================================================
def save_target_artifacts(result, save_path, run_label, num_fixed, total_electrodes,
                          fixed_names, num_classes, epoch):
    val_acc = result['val_acc']
    val_elec = result['val_elec']
    eval_acc = result['eval_acc']
    eval_bacc = result['eval_bacc']
    eval_prec = result['eval_prec']
    eval_rec = result['eval_rec']
    eval_f1 = result['eval_f1']
    eval_kap = result['eval_kap']

    eval_true_labels = result['eval_true']
    eval_pred_labels = result['eval_pred']

    train_loss_history = result['train_loss_hist']
    val_loss_history = result['val_loss_hist']
    train_acc_history = result['train_acc_hist']
    val_acc_history = result['val_acc_hist']

    class_gate_accumulation = result['class_gate_sum']
    class_sample_counts = result['class_counts']

    fixed_ratio = (num_fixed / total_electrodes) if total_electrodes > 0 else 0.0
    epochs_range = list(range(1, epoch + 1))

    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_loss_history, label='Training Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    plt.savefig(f'{save_path}loss_curve.png'); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_acc_history, label='Training Accuracy')
    plt.plot(epochs_range, val_acc_history, label='Validation Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend(); plt.grid(True)
    plt.savefig(f'{save_path}accuracy_curve.png'); plt.close()

    # 電極マップのプロット
    final_counts = class_sample_counts.unsqueeze(1)
    final_counts[final_counts == 0] = 1.0
    final_class_distribution = class_gate_accumulation / final_counts
    final_class_distribution = final_class_distribution.numpy()

    fig, axes = plt.subplots(1, num_classes, figsize=(18, 5))
    fig.suptitle(f'Electrode Selection (Total {total_electrodes}ch, Fixed {num_fixed}ch)', fontsize=16)

    global_vmin = 0.0
    global_vmax = 1.0

    for i in range(num_classes):
        ax = axes[i]
        state_name = CLASS_LABELS[i]
        usage = final_class_distribution[i]
        im = plot_head_map(ax, CHANNEL_NAMES, usage, vmin=global_vmin, vmax=global_vmax, title=state_name)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Selection Probability')

    plt.savefig(f'{save_path}class_wise_topography.png')
    plt.close()

    target_names = [CLASS_LABELS[i] for i in range(num_classes)]
    report = classification_report(eval_true_labels, eval_pred_labels, target_names=target_names, digits=4, zero_division=0)

    # 結果のテキスト保存
    with open(f'{save_path}holdout_eval_results.txt', 'w') as f:
        f.write("===== Hyperparameters =====\n")
        f.write(f"  lr_gate:            {LR_GATE}\n")
        f.write(f"  lr_classifier:      {LR_CLASSIFIER}\n")
        f.write(f"  weight_decay:       {WEIGHT_DECAY}\n")
        f.write(f"  batch_size:         {BATCH_SIZE}\n")
        f.write(f"  epoch:              {epoch}\n")
        f.write(f"  lambda_sparsity:    {LAMBDA_SPARSITY}\n")
        f.write(f"  total_electrodes:   {total_electrodes}\n")
        f.write(f"  fixed_electrodes:   {num_fixed} ({fixed_ratio*100:.1f}% of total)\n")
        f.write(f"  variable_target:    {total_electrodes - num_fixed}\n")
        f.write(f"  fixed_channels:     {fixed_names}\n")
        f.write(f"  model_selection:    Val Accuracy (best epoch)\n")
        f.write(f"  gate_feat:          {GATE_FEAT}\n")
        f.write(f"  gate_teacher:       {GATE_TEACHER}\n")
        f.write(f"  gate_hints:         {GATE_HINTS}\n")
        f.write(f"  select:             {SELECT}\n")
        f.write(f"  mask_randomize:     {MASK_RANDOMIZE}\n\n")

        f.write("===== Holdout Validation Results (at best Val-Acc epoch) =====\n")
        f.write(f"-> Val Accuracy: {val_acc:.4f}%\n")
        f.write(f"-> Avg Selected Electrodes: {val_elec:.2f}ch (Target: {total_electrodes}ch)\n\n")

        f.write("===== Final Eval (Test) Results =====\n")
        f.write(f"-> Eval Accuracy:  {eval_acc:.4f}%\n")
        f.write(f"-> Eval BACC:      {eval_bacc:.4f}\n")
        f.write(f"-> Eval Precision: {eval_prec:.4f}\n")
        f.write(f"-> Eval Recall:    {eval_rec:.4f}\n")
        f.write(f"-> Eval F1-Score:  {eval_f1:.4f}\n")
        f.write(f"-> Eval Kappa:     {eval_kap:.4f}\n\n")

        f.write("===== Eval Classification Report =====\n")
        f.write(report)
        f.write("\n")

        # === 電極の選定率を追記 ===
        f.write("===== Electrode Selection Rates =====\n")
        overall_usage = class_gate_accumulation.sum(dim=0) / class_sample_counts.sum()
        overall_usage = overall_usage.numpy()
        usage_ranking = [(CHANNEL_NAMES[i], overall_usage[i]) for i in range(NUM_CHANNELS)]
        usage_ranking.sort(key=lambda x: x[1], reverse=True)

        f.write("--- Overall Ranking ---\n")
        for rank, (ch, prob) in enumerate(usage_ranking, 1):
            fixed_mark = " (FIXED)" if ch in fixed_names else ""
            f.write(f"{rank:2d}. {ch:6s} : {prob:.4f}{fixed_mark}\n")

        f.write("\n--- Per Class Ranking ---\n")
        for c in range(num_classes):
            c_name = CLASS_LABELS[c]
            c_usage = final_class_distribution[c]
            c_ranking = [(CHANNEL_NAMES[i], c_usage[i]) for i in range(NUM_CHANNELS)]
            c_ranking.sort(key=lambda x: x[1], reverse=True)
            f.write(f"[{c_name}]\n")
            for rank, (ch, prob) in enumerate(c_ranking, 1):
                f.write(f"  {rank:2d}. {ch:6s} : {prob:.4f}\n")
            f.write("\n")

    # 混同行列の保存
    try:
        cm = confusion_matrix(eval_true_labels, eval_pred_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Eval Confusion Matrix (Acc: {eval_acc:.2f}%)')
        plt.savefig(f'{save_path}eval_confusion_matrix.png')
        plt.close(fig)
    except Exception as e:
        print(f"[{run_label}] Error saving confusion matrix: {e}", flush=True)

    print(f"[{run_label}] Metrics saved to {save_path}holdout_eval_results.txt", flush=True)


# =========================================================================
# ワーカー: 1つの「固定電極数」について学習・評価・成果物保存を行う
#   並列プール(mp.Pool)から呼び出される。GPUは worker_idx で割り当てる。
#
#   引数はすべて明示的に渡す (spawn では __main__ で設定した値が
#   ワーカー側のモジュール再importに引き継がれないため)。
# =========================================================================
def run_one_target(worker_idx, num_fixed, total_electrodes, epoch, base_dir, ranking,
                   X_train, y_train, X_val, y_val, X_eval, y_eval,
                   num_classes, num_gpus):
    # --- 使用GPUの決定 (固定電極数間で公平に分散) ---
    gpu_id = worker_idx % num_gpus if num_gpus > 0 else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # --- 固定電極数によらず同一シードで学習し、比較を公平にする ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- 固定/可変チャネルの決定 ---
    #   重要度ランキング上位 num_fixed 個を固定電極にする。
    fixed_names = ranking[:num_fixed]
    fixed_indices = [CHANNEL_NAMES.index(name) for name in fixed_names]
    variable_indices = [i for i in range(NUM_CHANNELS) if i not in fixed_indices]
    # 可変チャネルから追加で選ぶ数 = 総電極数 - 固定数
    target_var_electrodes = max(0, total_electrodes - num_fixed)

    run_label = f"Total{total_electrodes}-Fix{num_fixed}"
    save_path = os.path.join(base_dir, f'fix{num_fixed}/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"\n[{run_label}] GPU {gpu_id} | fixed={fixed_names} | "
          f"var_target={target_var_electrodes} | -> {save_path}", flush=True)

    # 失敗しても他の固定電極数の集計を止めないよう、例外はここで捕捉する
    try:
        result = train_eval_holdout(
            X_train, y_train, X_val, y_val, X_eval, y_eval,
            num_classes, NUM_CHANNELS, fixed_indices, variable_indices,
            BATCH_SIZE, epoch, target_var_electrodes, run_label,
            LAMBDA_SPARSITY, LR_GATE, LR_CLASSIFIER,
            PRETRAINED_MODEL_PATH, save_path, device
        )

        save_target_artifacts(result, save_path, run_label, num_fixed, total_electrodes,
                              fixed_names, num_classes, epoch)

        summary = {
            'n_total_electrodes': total_electrodes,
            'n_fixed_electrodes': num_fixed,
            'fixed_ratio': round(num_fixed / total_electrodes, 4) if total_electrodes > 0 else 0.0,
            'n_variable_target': target_var_electrodes,
            'fixed_channels': ';'.join(fixed_names),
            'avg_selected_electrodes': round(float(result['val_elec']), 4),
            'val_acc': round(float(result['val_acc']), 4),
            'eval_acc': round(float(result['eval_acc']), 4),
            'eval_bacc': round(float(result['eval_bacc']), 6),
            'eval_precision': round(float(result['eval_prec']), 6),
            'eval_recall': round(float(result['eval_rec']), 6),
            'eval_f1': round(float(result['eval_f1']), 6),
            'eval_kappa': round(float(result['eval_kap']), 6),
            'status': 'OK',
        }
        print(f"[{run_label}] DONE | "
              f"Eval BACC={summary['eval_bacc']:.4f} F1={summary['eval_f1']:.4f} "
              f"Kappa={summary['eval_kappa']:.4f} (avg {summary['avg_selected_electrodes']:.1f}ch)", flush=True)
        return summary

    except Exception as e:
        print(f"[{run_label}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return {
            'n_total_electrodes': total_electrodes,
            'n_fixed_electrodes': num_fixed,
            'fixed_ratio': round(num_fixed / total_electrodes, 4) if total_electrodes > 0 else 0.0,
            'n_variable_target': target_var_electrodes,
            'fixed_channels': ';'.join(fixed_names),
            'avg_selected_electrodes': float('nan'),
            'val_acc': float('nan'),
            'eval_acc': float('nan'),
            'eval_bacc': float('nan'),
            'eval_precision': float('nan'),
            'eval_recall': float('nan'),
            'eval_f1': float('nan'),
            'eval_kappa': float('nan'),
            'status': f'FAILED: {e}',
        }


# =========================================================================
# メイン処理:
#   総電極数を指定 -> results_channel{N}/ を作成 -> 固定電極数 0..N を
#   並列に評価して CSV に集約し、比較図を作成する。
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='固定電極の割合 (0〜総電極数) をスイープして比較する実験')
    parser.add_argument('--total', type=int, default=DEFAULT_TOTAL_ELECTRODES,
                        help=f'総電極数 (1〜{NUM_CHANNELS})。既定={DEFAULT_TOTAL_ELECTRODES}')
    parser.add_argument('--epoch', type=int, default=DEFAULT_EPOCH,
                        help=f'学習エポック数。既定={DEFAULT_EPOCH}')
    args = parser.parse_args()

    TOTAL_ELECTRODES = args.total
    EPOCH = args.epoch

    if not (1 <= TOTAL_ELECTRODES <= NUM_CHANNELS):
        raise ValueError(f"--total は 1〜{NUM_CHANNELS} を指定してください (指定値: {TOTAL_ELECTRODES})")

    # ① 指定した総電極数に応じた出力フォルダを作成
    base_dir = os.path.join(RESEARCH_ROOT_DIR, f'results_channel{TOTAL_ELECTRODES}/')
    summary_csv_path = os.path.join(base_dir, 'summary_fix_ratio.csv')

    mp.set_start_method('spawn', force=True)

    def set_seed(s):
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(SEED)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # ② 事前学習の重要度ランキング (コード埋め込み) を固定電極の選定順に使う
    ranking = IMPORTANCE_RANKING

    # 固定電極数のスイープ範囲: 0 (固定なし) 〜 総電極数 (全固定)
    FIX_COUNTS = list(range(0, TOTAL_ELECTRODES + 1))

    print(f"Total electrodes (fixed target): {TOTAL_ELECTRODES}")
    print(f"Importance ranking (high -> low): {ranking}")
    print(f"Sweeping number of FIXED electrodes: {FIX_COUNTS}")
    print(f"Output base dir: {base_dir}")

    # --- データは1回だけ読み込み、共有メモリで各プロセスに渡す ---
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes = load_data()
    print(f"Number of classes (TUEV): {num_classes}")

    X_train.share_memory_(); y_train.share_memory_()
    X_val.share_memory_();   y_val.share_memory_()
    X_eval.share_memory_();  y_eval.share_memory_()

    num_gpus = torch.cuda.device_count()

    # --- 並列度: GPU数 × PROCESSES_PER_GPU (固定電極数を超えない範囲で) ---
    if num_gpus > 0:
        num_workers = min(len(FIX_COUNTS), max(1, num_gpus * PROCESSES_PER_GPU))
    else:
        num_workers = 1  # CPU実行時は逐次
    print(f"Detected {num_gpus} GPU(s) -> running {num_workers} parallel worker(s)")

    args_list = []
    for worker_idx, num_fixed in enumerate(FIX_COUNTS):
        args_list.append((worker_idx, num_fixed, TOTAL_ELECTRODES, EPOCH, base_dir, ranking,
                          X_train, y_train, X_val, y_val, X_eval, y_eval,
                          num_classes, num_gpus))

    print(f"\n--- Starting parallel sweep over {len(FIX_COUNTS)} fixed-electrode counts ---")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(run_one_target, args_list, chunksize=1)

    # 固定電極数の昇順に整列
    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x['n_fixed_electrodes'])

    # =========================================================================
    # 結果を CSV に集約 (DBは作らない)
    # =========================================================================
    fieldnames = [
        'n_total_electrodes', 'n_fixed_electrodes', 'fixed_ratio', 'n_variable_target',
        'fixed_channels', 'avg_selected_electrodes',
        'val_acc', 'eval_acc', 'eval_bacc',
        'eval_precision', 'eval_recall', 'eval_f1', 'eval_kappa',
        'status',
    ]
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nSummary CSV saved to: {summary_csv_path}")

    # --- 標準出力にも一覧表示 ---
    print("\n===== Summary (sorted by number of fixed electrodes) =====")
    print(f"{'Fix':>3} | {'ratio':>5} | {'avgCh':>6} | {'evalBACC':>8} | {'evalF1':>7} | "
          f"{'evalKappa':>9} | {'evalAcc':>7} | status")
    print("-" * 82)
    for r in results:
        print(f"{r['n_fixed_electrodes']:>3} | {r['fixed_ratio']:>5.2f} | "
              f"{r['avg_selected_electrodes']:>6.2f} | {r['eval_bacc']:>8.4f} | {r['eval_f1']:>7.4f} | "
              f"{r['eval_kappa']:>9.4f} | {r['eval_acc']:>7.2f} | {r['status']}")

    # --- 最良の固定電極数 (Eval BACC 基準) を報告 ---
    ok_results = [r for r in results if r['status'] == 'OK']
    if ok_results:
        best = max(ok_results, key=lambda x: x['eval_bacc'])
        print(f"\n>>> Best by Eval Balanced Accuracy: "
              f"Fixed {best['n_fixed_electrodes']}ch "
              f"(ratio={best['fixed_ratio']:.2f}, BACC={best['eval_bacc']:.4f}, "
              f"F1={best['eval_f1']:.4f}, Kappa={best['eval_kappa']:.4f})")

    # =========================================================================
    # 固定電極数 vs 各指標 の比較プロット (どの固定割合が良いか一目で分かるように)
    # =========================================================================
    if ok_results:
        ns = [r['n_fixed_electrodes'] for r in ok_results]
        plt.figure(figsize=(10, 6))
        plt.plot(ns, [r['eval_bacc'] for r in ok_results], marker='o', label='Balanced Accuracy')
        plt.plot(ns, [r['eval_f1'] for r in ok_results], marker='s', label='Macro F1')
        plt.plot(ns, [r['eval_kappa'] for r in ok_results], marker='^', label="Cohen's Kappa")
        plt.plot(ns, [r['eval_acc'] / 100.0 for r in ok_results], marker='d', label='Accuracy (/100)')
        plt.xlabel(f'Number of FIXED electrodes (out of total {TOTAL_ELECTRODES})')
        plt.ylabel('Score')
        plt.title(f'Evaluation metrics vs. number of fixed electrodes (Total {TOTAL_ELECTRODES}ch)')
        plt.xticks(FIX_COUNTS)
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(base_dir, 'summary_metrics_vs_fixed.png'),
                    bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved to: {os.path.join(base_dir, 'summary_metrics_vs_fixed.png')}")

    print("\nAll fixed-electrode-ratio experiments finished!")
