"""
paper_2class (SEED-VLA) データを用いた「固定電極 / 動的電極の割合」の探索 (EEG 駆動のゲート)。

【ベースライン版 + グローバル標準化 + 事前キャッシュ読み込み専用】
総電極数 TOTAL のうち何本を「固定電極」、何本を「動的電極」にするかを探索する
コアロジックはそのまま維持し、AdaBN、Sampler、平滑化などの追加処理を撤廃した
純粋なベースラインコードです。

データリークを防ぐためのグローバル標準化方式を採用しています。
このスクリプトは自ら教師モデルを回してキャッシュを生成せず、
指定されたディレクトリの事前キャッシュ (.npz) を利用して学習を行います。
"""

import argparse
import csv
import glob
import json
import os
import sys
import traceback

# プロジェクトルートを import パスに追加 (model を解決するため)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use('Agg')  # 並列プロセス / ヘッドレス環境で描画するため非GUIバックエンド
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import warnings
from scipy.interpolate import griddata
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

from model.EEGNet import CustomEEGNet

warnings.filterwarnings("ignore")

# =========================================================================
# 設定
# =========================================================================
DATA_DIR = '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA/processdData/paper_2class'

# キャッシュ名解決用のデフォルトタグ（キャッシュ検索時のフォールバック用）
CHEAT_SHEET_DIR = os.environ.get(
    'CHEAT_SHEET_DIR',
    '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA/'
    'LOSO_pretrainedmodel/LOSO_Baseline_best')

CHEAT_SHEET_TAG = '_'.join(
    [p for p in CHEAT_SHEET_DIR.rstrip('/').split('/')[-2:] if p])

RESEARCH_BASE_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA/'
                     'research_ratio_fixdynamic/'
                     'results_ratio/')

NUM_CLASSES = 2
CLASS_NAMES = ['Awake', 'Fatigue']
CLASS_LABELS = {0: 'Awake', 1: 'Fatigue'}

# 10-20系18ch。
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T3', 'C3', 'Cz', 'C4', 'T4',
                 'T5', 'P3', 'T6', 'P4', 'O1', 'O2']

NUM_CHANNELS = len(CHANNEL_NAMES)

# DE特徴の周波数帯数。入力 [N, 16, 5, 18] の 3次元目。
NUM_BANDS = 5

DEFAULT_TOTAL_ELECTRODES = 10

# --ranking global 用の共通ランキング (重要度が高い順)
GLOBAL_IMPORTANCE_RANKING = [ 'T5', 'O1', 'O2', 'F8', 'P4', 'F7', 'T4', 'C3', 'T3', 'Fp1', 'Fz', 'Fp2', 'T6', 'Cz', 'C4', 'F3', 'F4', 'P3' ]

SEED = 42
BATCH_SIZE = 128

LR_GATE = 1e-4
LR_CLASSIFIER = 2e-4
WEIGHT_DECAY = 0.03
LAMBDA_SPARSITY = 1e-3   # --select threshold のときだけ効く (topk では罰則なし)

FIXED_EPOCHS = 100

# --- クラス重み ---
CLASS_WEIGHT = 'none'

# --- Gate の設計 ---
GATE_FEAT = 'band'          # Gate に渡す EEG 特徴。band: 帯域を残して [B,5,18]->90次元
GATE_TEACHER = 'fold_const' # 教師の電極重要度の入れ方
GATE_HINTS = False          # 教師の予測確率を Gate に入れるか
SELECT = 'topk'             # 可変チャネルからスコア上位 (total-fixed) 本を厳密に選ぶ
MASK_RANDOMIZE = 0.3        # 学習時にこの確率で可変チャネルのマスクをランダムに差し替える

# 1つのGPUに載せる並列プロセス数
PROCESSES_PER_GPU = 4

# 報告する評価指標
REPORT_METRICS = [('acc', 'Accuracy '),
                  ('macro_precision', 'Precision'),
                  ('macro_recall', 'Recall   '),
                  ('macro_f1', 'F1-Score ')]
PCT_METRICS = {'acc'}

# 頭部トポマップ上のおおよその位置
COORDS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.8, 0.6), 'F3': (-0.4, 0.6), 'Fz': (0, 0.6), 'F4': (0.4, 0.6), 'F8': (0.8, 0.6),
    'T3': (-0.9, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0, 0.0), 'C4': (0.5, 0.0), 'T4': (0.9, 0.0),
    'T5': (-0.8, -0.6), 'P3': (-0.4, -0.6), 'P4': (0.4, -0.6), 'T6': (0.8, -0.6),
    'O1': (-0.3, -0.9), 'O2': (0.3, -0.9),
}


# =========================================================================
# 0. データ
# =========================================================================
def load_subject_x(subject_id):
    """1被験者分の生のDE特徴を読む。"""
    x = np.load(os.path.join(DATA_DIR, f'paper_eeg_{subject_id}.npy')).astype(np.float32)
    return x


def load_subjects(subject_ids):
    """paper_2class から指定被験者を読み込んで連結する（標準化はここでは行わない）。"""
    xs, ys = [], []
    for s in subject_ids:
        xs.append(load_subject_x(s))
        ys.append(np.load(os.path.join(DATA_DIR, f'paper_label_{s}.npy')))
    x = torch.FloatTensor(np.concatenate(xs, axis=0))
    y = torch.LongTensor(np.concatenate(ys, axis=0).astype(np.int64))
    return x, y


def class_weights(subject_ids, mode, device):
    if mode == 'none':
        return None
    y = np.concatenate([np.load(os.path.join(DATA_DIR, f'paper_label_{s}.npy'))
                        for s in subject_ids])
    n = len(y)
    counts = np.array([(y == c).sum() for c in range(NUM_CLASSES)], dtype=np.float64)
    if mode == 'balanced':
        w = np.where(counts > 0, n / (NUM_CLASSES * np.maximum(counts, 1)), 1.0)
        return torch.FloatTensor(w).to(device)
    raise ValueError(f'unknown class_weight mode: {mode}')


def fold_constant_importance(teacher_cache, train_subjects):
    """fold 内で共通に使う教師の電極重要度 [18] を、【学習被験者だけ】から作る。"""
    return np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects],
                          axis=0).mean(axis=0)


def make_loader(subject_ids, shuffle, teacher_cache,
                imp_override=None, generator=None, train_mean=None, train_std=None):
    """入力・ラベルに加えて、キャッシュ済みの教師出力も一緒に流すローダ。
       Source 1方式のグローバル標準化（Trainデータの平均/分散）を適用する。
    """
    x, y = load_subjects(subject_ids)
    
    if train_mean is not None and train_std is not None:
        x = (x - train_mean) / (train_std + 1e-8)
        
    hints = torch.FloatTensor(
        np.concatenate([teacher_cache[f'hints_{s}'] for s in subject_ids], axis=0))
    if imp_override is None:
        imp = torch.FloatTensor(
            np.concatenate([teacher_cache[f'imp_{s}'] for s in subject_ids], axis=0))
    else:
        imp = torch.FloatTensor(
            np.repeat(np.asarray(imp_override, dtype=np.float32)[None, :], len(x), axis=0))
    
    assert len(hints) == len(x) and len(imp) == len(x), '教師キャッシュとサンプル数が不一致'
    ds = TensorDataset(x, y, hints, imp)
    
    # Sampler処理は削除し、通常のランダムサンプリングを使用
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


# =========================================================================
# 1. Gate Mechanism & Cache Path Resolution
# =========================================================================
def teacher_cache_path(cache_dir, mode, target):
    """
    指定されたディレクトリから対象のキャッシュファイルを見つける。
    完全一致がない場合は、柔軟にパターンマッチで検索する。
    """
    exact_path = os.path.join(cache_dir, f'teacher_{mode}_{CHEAT_SHEET_TAG}_globalnorm_target{target}.npz')
    if os.path.exists(exact_path):
        return exact_path
        
    # 見つからない場合はパターンで検索
    pattern = os.path.join(cache_dir, f'teacher_{mode}_*_target{target}.npz')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"指定されたディレクトリにターゲット{target}のキャッシュが見つかりません: {pattern}")
    
    # 複数見つかった場合は globalnorm が含まれるものを優先
    if len(matches) > 1:
        gn_matches = [m for m in matches if 'globalnorm' in m]
        if gn_matches:
            return gn_matches[0]
    return matches[0]


class GateMechanism(nn.Module):
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
# =========================================================================
class ConceptDynamicNet(nn.Module):
    def __init__(self, classifier_model, num_channels=18,
                 fixed_indices=[], variable_indices=[], num_classes=2,
                 n_select_var=None, gate_feat=GATE_FEAT, use_hints=GATE_HINTS,
                 imp_dim=NUM_CHANNELS, select=SELECT, mask_randomize=0.0,
                 n_bands=5):
        super(ConceptDynamicNet, self).__init__()

        self.fixed_indices = fixed_indices
        self.variable_indices = variable_indices
        self.num_channels = num_channels
        self.gate_feat = gate_feat
        self.use_hints = use_hints
        self.imp_dim = imp_dim
        self.select = select
        self.mask_randomize = mask_randomize
        self.n_select_var = n_select_var

        feat_dim = num_channels * n_bands if gate_feat == 'band' else num_channels
        self.gate = GateMechanism(feat_dim=feat_dim,
                                  num_variable_channels=len(variable_indices),
                                  hint_dim=num_classes if use_hints else 0,
                                  imp_dim=imp_dim)
        self.classifier = classifier_model

    def gate_features(self, x):
        if self.gate_feat == 'band':
            return x.mean(dim=1).reshape(x.size(0), -1)
        return x.mean(dim=(1, 2))

    def _select(self, scores):
        k = self.n_select_var
        n_var = scores.size(1)
        if self.select == 'topk' and k is not None:
            if k <= 0:
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
        k = self.n_select_var
        n_var = like.size(1)
        if k is None or k >= n_var:
            return torch.ones_like(like)
        if k <= 0:
            return torch.zeros_like(like)
        idx = torch.rand_like(like).topk(k, dim=1).indices
        return torch.zeros_like(like).scatter(1, idx, 1.0)

    def forward(self, x, teacher_hints, teacher_importance):
        batch_size = x.size(0)
        device = x.device

        full_gate_weights = torch.zeros(batch_size, self.num_channels, device=device)
        if len(self.fixed_indices) > 0:
            full_gate_weights[:, self.fixed_indices] = 1.0

        applied = full_gate_weights
        if len(self.variable_indices) > 0:
            feat = self.gate_features(x)
            scores = self.gate(feat,
                               teacher_hints if self.use_hints else None,
                               teacher_importance if self.imp_dim > 0 else None)
            gate_weights_var = self._select(scores)
            full_gate_weights[:, self.variable_indices] = gate_weights_var

            if self.training and self.mask_randomize > 0:
                swap = torch.rand(batch_size, device=device) < self.mask_randomize
                if bool(swap.any()):
                    rnd_var = self._random_mask(gate_weights_var.detach())
                    rnd_full = torch.zeros(batch_size, self.num_channels, device=device)
                    if len(self.fixed_indices) > 0:
                        rnd_full[:, self.fixed_indices] = 1.0
                    rnd_full[:, self.variable_indices] = rnd_var
                    applied = torch.where(swap.unsqueeze(1), rnd_full, full_gate_weights)
        else:
            gate_weights_var = torch.zeros(batch_size, 0, device=device)

        x_masked = x * applied.unsqueeze(1).unsqueeze(1)
        outputs = self.classifier(x_masked)

        return outputs, gate_weights_var, full_gate_weights, teacher_importance


# =========================================================================
# 3. 固定電極の選び方 (重要度ランキング)
# =========================================================================
def fold_ranking(teacher_cache, train_subjects):
    """そのfoldの学習被験者のサンプルだけから電極重要度ランキングを作る。"""
    imp = np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects],
                         axis=0).mean(axis=0)                      # [18]
    order = np.argsort(imp)[::-1]
    return [CHANNEL_NAMES[i] for i in order], imp


def resolve_ranking(mode, teacher_cache, train_subjects):
    """--ranking の指定に応じて、固定電極を選ぶ順序を返す。"""
    if mode == 'global':
        return list(GLOBAL_IMPORTANCE_RANKING), None
    return fold_ranking(teacher_cache, train_subjects)


# =========================================================================
# 4. 可視化
# =========================================================================
def plot_head_map(ax, channel_names, usage_weights, title=""):
    x_coords, y_coords, z_values = [], [], []
    for i, name in enumerate(channel_names):
        if name in COORDS:
            x, y = COORDS[name]
            x_coords.append(x)
            y_coords.append(y)
            z_values.append(usage_weights[i])

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_values = np.array(z_values)

    angles = np.linspace(0, 2 * np.pi, 36)
    edge_x, edge_y = np.cos(angles), np.sin(angles)
    edge_z = []
    for ex, ey in zip(edge_x, edge_y):
        dists = np.sqrt((x_coords - ex) ** 2 + (y_coords - ey) ** 2)
        edge_z.append(z_values[np.argmin(dists)])

    x_coords = np.concatenate([x_coords, edge_x])
    y_coords = np.concatenate([y_coords, edge_y])
    z_values = np.concatenate([z_values, np.array(edge_z)])

    grid_x, grid_y = np.mgrid[-1.2:1.2:300j, -1.2:1.2:300j]
    grid_z = griddata((x_coords, y_coords), z_values, (grid_x, grid_y),
                      method='cubic', fill_value=0.0)
    grid_z = np.clip(grid_z, 0.0, 1.0)

    dist = np.sqrt(grid_x ** 2 + grid_y ** 2)
    grid_z = np.ma.masked_where(dist > 1.0, grid_z)

    levels = np.linspace(0.0, 1.0, 100)
    im = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='jet', vmin=0.0, vmax=1.0)

    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, linewidth=2))
    ax.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k', linewidth=2)

    for i, name in enumerate(channel_names):
        if name in COORDS:
            cx, cy = COORDS[name]
            ax.scatter(cx, cy, c='k', s=18, marker='o', alpha=0.5, zorder=5)
            ax.text(cx, cy + 0.075, name, fontsize=7.5, ha='center', va='center',
                    color='black', alpha=0.85, zorder=6)

    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.4)
    ax.set_aspect('equal')
    return im


def plot_class_topography(class_distribution, out_path, total, num_fixed, n_folds):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(6 * NUM_CLASSES, 5.5))
    fig.suptitle(f'Electrode Selection (Total {total}ch / Fixed {num_fixed}ch, '
                 f'mean of {n_folds} LOSO folds)', fontsize=15)
    im = None
    for c in range(NUM_CLASSES):
        im = plot_head_map(axes[c], CHANNEL_NAMES, class_distribution[c],
                           title=CLASS_LABELS[c])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label('Selection Probability')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_selection_bar(usage, fixed_rate, out_path, total, num_fixed, n_folds):
    order = np.argsort(usage)[::-1]
    names = [CHANNEL_NAMES[i] for i in order]
    plt.figure(figsize=(11, 6))
    sns.barplot(x=names, y=[usage[i] for i in order], palette="Reds_r")
    if fixed_rate is not None:
        plt.plot(range(len(order)), [fixed_rate[i] for i in order],
                 'k^--', markersize=6, linewidth=1, label='Fixed rate across folds')
        plt.legend()
    plt.title(f'Electrode Selection Rate (Total {total}ch / Fixed {num_fixed}ch, '
              f'{n_folds} LOSO folds)')
    plt.xlabel("Channel"); plt.ylabel("Selection Probability")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(trues, preds, out_path, title):
    cm = confusion_matrix(trues, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=CLASS_NAMES).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================================
# 5. 1 fold (固定数 x ターゲット被験者) の学習・評価
# =========================================================================
@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, n_batch = 0.0, 0
    trues, preds = [], []

    gate_sum = torch.zeros(NUM_CLASSES, NUM_CHANNELS)
    gate_counts = torch.zeros(NUM_CLASSES)
    usage_sum, n_seen = 0.0, 0

    for x, y, hints, imp in loader:
        if x.size(0) == 0:
            continue
        x, y = x.to(device), y.to(device).view(-1)
        hints, imp = hints.to(device), imp.to(device)
        logits, _, full_gate_weights, _ = model(x, hints, imp)
        total_loss += loss_fn(logits, y).item()
        n_batch += 1

        pred = logits.argmax(dim=1)
        trues.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())

        n_seen += x.size(0)
        usage_sum += full_gate_weights.sum().item()
        gates = full_gate_weights.detach().cpu()
        y_cpu = y.cpu()
        for c in range(NUM_CLASSES):
            mask = (y_cpu == c)
            if mask.sum() > 0:
                gate_sum[c] += gates[mask].sum(dim=0)
                gate_counts[c] += mask.sum()

    labels = list(range(NUM_CLASSES))
    metrics = {
        'loss': total_loss / max(n_batch, 1),
        'acc': 100.0 * accuracy_score(trues, preds),
        'macro_precision': precision_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_recall': recall_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_f1': f1_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'avg_selected_electrodes': usage_sum / max(n_seen, 1),
        'n_class0_true': int(sum(1 for t in trues if t == 0)),
        'n_class1_true': int(sum(1 for t in trues if t == 1)),
        'both_classes_present': len(set(trues)) == NUM_CLASSES,
    }
    return metrics, trues, preds, gate_sum, gate_counts


def run_fold(total, num_fixed, target, kept, epochs, teacher_importance, ranking_mode,
             full_fix_mode, device, fix_dir, cache_dir, class_weight=CLASS_WEIGHT,
             gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER, gate_hints=GATE_HINTS,
             select=SELECT, mask_randomize=MASK_RANDOMIZE):
    
    train_subjects = [s for s in kept if s != target]

    seed = SEED + target
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)

    tag = f'[T{total}/F{num_fixed}/Sub{target}]'

    # --- Trainデータ全体から mean / std を計算 (Source 1 方式) ---
    x_train_raw, _ = load_subjects(train_subjects)
    train_mean = x_train_raw.mean(dim=0, keepdim=True)
    train_std = x_train_raw.std(dim=0, keepdim=True)
    del x_train_raw
    # -------------------------------------------------------------------

    # キャッシュを読み込む (生成処理は行わない)
    cache_path = teacher_cache_path(cache_dir, teacher_importance, target)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"教師キャッシュが見つかりません: {cache_path}")
    teacher_cache = np.load(cache_path)

    if gate_teacher == 'fold_const':
        imp_override = fold_constant_importance(teacher_cache, train_subjects)
        imp_dim = NUM_CHANNELS
    elif gate_teacher == 'per_sample':
        imp_override, imp_dim = None, NUM_CHANNELS
    elif gate_teacher == 'none':
        imp_override, imp_dim = None, 0
    else:
        raise ValueError(f'unknown gate_teacher: {gate_teacher}')

    # --- 固定/可変チャネルの決定 ---
    ranking, ranking_scores = resolve_ranking(ranking_mode, teacher_cache, train_subjects)
    fixed_names = ranking[:num_fixed]
    fixed_indices = [CHANNEL_NAMES.index(n) for n in fixed_names]
    target_var_electrodes = max(0, total - num_fixed)

    if num_fixed >= total and full_fix_mode == 'static':
        variable_indices = []
    else:
        variable_indices = [i for i in range(NUM_CHANNELS) if i not in fixed_indices]

    print(f"{tag} train {len(train_subjects)}名 {train_subjects} / test [{target}] "
          f"| fixed={fixed_names} | var_target={target_var_electrodes}"
          f"/{len(variable_indices)} | epochs={epochs} "
          f"| global_norm=True | class_weight={class_weight} "
          f"| gate_feat={gate_feat} | gate_teacher={gate_teacher} | gate_hints={gate_hints} "
          f"| select={select} | mask_rand={mask_randomize} | {device}", flush=True)

    classifier_model = CustomEEGNet(numclasses=NUM_CLASSES).to(device)

    model = ConceptDynamicNet(
        classifier_model,
        num_channels=NUM_CHANNELS,
        fixed_indices=fixed_indices,
        variable_indices=variable_indices,
        num_classes=NUM_CLASSES,
        n_select_var=target_var_electrodes,
        gate_feat=gate_feat,
        use_hints=gate_hints,
        imp_dim=imp_dim,
        select=select,
        mask_randomize=mask_randomize,
        n_bands=NUM_BANDS,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.gate.parameters(), 'lr': LR_GATE},
        {'params': model.classifier.parameters(), 'lr': LR_CLASSIFIER},
    ], weight_decay=WEIGHT_DECAY)
    
    weights = class_weights(train_subjects, class_weight, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_loader = make_loader(train_subjects, shuffle=True, teacher_cache=teacher_cache,
                               imp_override=imp_override, generator=gen,
                               train_mean=train_mean, train_std=train_std)

    curve_train_loss, curve_train_acc, curve_usage = [], [], []
    for ep in range(epochs):
        model.train()
        run_loss, correct, seen, usage_sum = 0.0, 0, 0, 0.0

        for x, y, hints, imp in train_loader:
            x, y = x.to(device), y.to(device).view(-1)
            hints, imp = hints.to(device), imp.to(device)
            logits, gate_weights_var, full_gate_weights, _ = model(x, hints, imp)

            loss_cls = loss_fn(logits, y)
            if select == 'topk':
                loss = loss_cls
            else:
                usage_per_sample_var = gate_weights_var.sum(dim=1)
                loss_sparsity = ((usage_per_sample_var - target_var_electrodes) ** 2).mean()
                loss = loss_cls + (LAMBDA_SPARSITY * loss_sparsity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            usage_sum += full_gate_weights.sum().item()

        curve_train_loss.append(run_loss / len(train_loader))
        curve_train_acc.append(100.0 * correct / seen)
        curve_usage.append(usage_sum / seen)
        print(f"    {tag} Epoch {ep+1:3d}/{epochs} | loss {curve_train_loss[-1]:.4f} "
              f"| acc {curve_train_acc[-1]:5.2f}% | usage {curve_usage[-1]:.1f}ch", flush=True)

    test_loader = make_loader([target], shuffle=False, teacher_cache=teacher_cache,
                              imp_override=imp_override,
                              train_mean=train_mean, train_std=train_std)
    tm, trues, preds, gate_sum, gate_counts = evaluate(model, test_loader, device, loss_fn)

    print(f"{tag} test 覚醒{tm['n_class0_true']}/疲労{tm['n_class1_true']}"
          f"{'' if tm['both_classes_present'] else ' ※単一クラスのため指標が退化'} -> "
          f"Acc {tm['acc']:.2f}% | Prec {tm['macro_precision']:.4f} "
          f"| Rec {tm['macro_recall']:.4f} | F1 {tm['macro_f1']:.4f} "
          f"| 選定 {tm['avg_selected_electrodes']:.2f}ch (目標 {total}ch)", flush=True)

    result = {**tm, 'target': target,
              'n_total_electrodes': total, 'n_fixed_electrodes': num_fixed,
              'fixed_ratio': round(num_fixed / total, 4) if total > 0 else 0.0,
              'n_variable_target': target_var_electrodes,
              'n_variable_channels': len(variable_indices),
              'fixed_channels': fixed_names, 'ranking_mode': ranking_mode,
              'full_fix_mode': full_fix_mode,
              'n_test': len(trues), 'train_subjects': train_subjects, 'epochs': epochs,
              'teacher_importance': teacher_importance,
              'class_weight': class_weight,
              'gate_feat': gate_feat, 'gate_teacher': gate_teacher,
              'gate_hints': gate_hints, 'select': select,
              'mask_randomize': mask_randomize}

    with open(os.path.join(fix_dir, f'fold_target{target}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(fix_dir, f'fold_target{target}.npz'),
             trues=np.array(trues), preds=np.array(preds),
             gate_sum=gate_sum.numpy(), gate_counts=gate_counts.numpy(),
             fixed_indices=np.array(fixed_indices, dtype=np.int64),
             ranking_scores=(np.array(ranking_scores) if ranking_scores is not None
                             else np.zeros(NUM_CHANNELS)),
             curve_train_loss=np.array(curve_train_loss),
             curve_train_acc=np.array(curve_train_acc),
             curve_usage=np.array(curve_usage))
    torch.save(model.state_dict(),
               os.path.join(fix_dir, f'concept_dynamic_net_target{target}.pth'))

    del model, classifier_model, optimizer
    torch.cuda.empty_cache()
    return result


# =========================================================================
# 6. ワーカー (mp.Pool から呼ばれる)
# =========================================================================
def worker(job_idx, total, num_fixed, target, kept, epochs, teacher_importance,
           ranking_mode, full_fix_mode, num_gpus, base_dir, class_weight=CLASS_WEIGHT,
           gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER, gate_hints=GATE_HINTS,
           select=SELECT, mask_randomize=MASK_RANDOMIZE, cache_dir=None):
    gpu_id = job_idx % num_gpus if num_gpus > 0 else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    fix_dir = os.path.join(base_dir, f'total{total}', f'fix{num_fixed}')
    os.makedirs(fix_dir, exist_ok=True)

    try:
        return run_fold(total, num_fixed, target, kept, epochs, teacher_importance,
                        ranking_mode, full_fix_mode, device, fix_dir, cache_dir,
                        class_weight, gate_feat, gate_teacher, gate_hints, select, mask_randomize)
    except Exception as e:
        print(f"[T{total}/F{num_fixed}/Sub{target}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return None


# =========================================================================
# 7. 集約
# =========================================================================
def summarize(fold_results, subset):
    sub = [fold_results[s] for s in subset if s in fold_results]
    if not sub:
        return None
    out = {'n_subjects': len(sub), 'subjects': list(subset)}
    for key, _ in REPORT_METRICS:
        v = np.array([r[key] for r in sub], dtype=float)
        v = v[~np.isnan(v)]
        out[key] = {'mean': float(v.mean()), 'std': float(v.std())} if len(v) else None
    sel = np.array([r['avg_selected_electrodes'] for r in sub], dtype=float)
    out['avg_selected_electrodes'] = {'mean': float(sel.mean()), 'std': float(sel.std())}
    return out


def aggregate_fix(total, num_fixed, base_dir, manifest, verbose=True):
    fix_dir = os.path.join(base_dir, f'total{total}', f'fix{num_fixed}')
    files = sorted(glob.glob(os.path.join(fix_dir, 'fold_target*.json')))
    if not files:
        return None

    balanced = manifest['loso_target_subjects']

    fold_results, arrays = {}, {}
    for p in files:
        r = json.load(open(p))
        fold_results[r['target']] = r
        arrays[r['target']] = np.load(p.replace('.json', '.npz'))
    targets = sorted(fold_results)
    degenerate = [s for s in targets if s not in balanced]

    global_true = np.concatenate([arrays[t]['trues'] for t in targets]).tolist()
    global_pred = np.concatenate([arrays[t]['preds'] for t in targets]).tolist()

    summary_all = summarize(fold_results, targets)
    summary_balanced = summarize(fold_results, [s for s in targets if s in balanced])

    pooled = {
        'accuracy': accuracy_score(global_true, global_pred),
        'macro_precision': precision_score(global_true, global_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(global_true, global_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(global_true, global_pred, average='macro', zero_division=0),
    }
    report = classification_report(global_true, global_pred,
                                   target_names=CLASS_NAMES, digits=4, zero_division=0)

    gate_sum = np.sum([arrays[t]['gate_sum'] for t in targets], axis=0)        # [2, 18]
    gate_counts = np.sum([arrays[t]['gate_counts'] for t in targets], axis=0)  # [2]
    denom = gate_counts.copy()
    denom[denom == 0] = 1.0
    class_distribution = gate_sum / denom[:, None]
    overall_usage = gate_sum.sum(axis=0) / max(gate_counts.sum(), 1.0)

    fixed_rate = np.zeros(NUM_CHANNELS)
    for t in targets:
        fixed_rate[arrays[t]['fixed_indices'].astype(int)] += 1.0
    fixed_rate /= max(len(targets), 1)

    plot_confusion(global_true, global_pred,
                   os.path.join(fix_dir, 'confusion_matrix_pooled.png'),
                   f'Total {total}ch / Fixed {num_fixed}ch pooled LOSO '
                   f'(Acc: {pooled["accuracy"]*100:.2f}%, F1: {pooled["macro_f1"]:.4f})')
    plot_class_topography(class_distribution,
                          os.path.join(fix_dir, 'class_wise_topography.png'),
                          total, num_fixed, len(targets))
    plot_selection_bar(overall_usage, fixed_rate,
                       os.path.join(fix_dir, 'electrode_selection_rate.png'),
                       total, num_fixed, len(targets))

    try:
        n_ep = min(len(arrays[t]['curve_train_loss']) for t in targets)
        tl = np.stack([arrays[t]['curve_train_loss'][:n_ep] for t in targets])
        ep_axis = np.arange(1, n_ep + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ep_axis, tl.mean(axis=0), label='Train Loss', color='tab:blue')
        ax.fill_between(ep_axis, tl.mean(axis=0) - tl.std(axis=0),
                        tl.mean(axis=0) + tl.std(axis=0), color='tab:blue', alpha=0.2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss')
        ax.set_title(f'Total {total}ch / Fixed {num_fixed}ch LOSO average learning curve '
                     f'({len(targets)} folds, {n_ep} epochs fixed)')
        ax.legend(); fig.tight_layout()
        plt.savefig(os.path.join(fix_dir, 'learning_curve.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[T{total}/F{num_fixed}] Plotting Error: {e}")

    ratio = (num_fixed / total) if total > 0 else 0.0
    with open(os.path.join(fix_dir, 'loso_report.txt'), 'w') as f:
        f.write(f"===== LOSO 2-class (paper_2class / SEED-VLA Baseline) / total {total}ch, "
                f"fixed {num_fixed}ch ({ratio*100:.1f}%) =====\n")
        f.write("metrics: Accuracy / Precision / Recall / F1-Score "
                "(Precision, Recall, F1 は macro 平均)\n")
        f.write(f"epochs : {fold_results[targets[0]].get('epochs')} (全fold固定)\n")
        f.write(f"ranking: {fold_results[targets[0]].get('ranking_mode')} "
                f"(fold=学習被験者のみから作成 / global=全fold共通)\n")
        f.write(f"labeling: {manifest['labeling_rule']}\n")
        f.write("\n===== Hyperparameters =====\n")
        f.write(f"  lr_gate:           {LR_GATE}\n")
        f.write(f"  lr_classifier:     {LR_CLASSIFIER}\n")
        f.write(f"  weight_decay:      {WEIGHT_DECAY}\n")
        f.write(f"  batch_size:        {BATCH_SIZE}\n")
        f.write(f"  lambda_sparsity:   {LAMBDA_SPARSITY} "
                f"(select=threshold のときだけ有効)\n")
        r0 = fold_results[targets[0]]
        f.write(f"  normalization:     Global Norm (based on Train set only)\n")
        f.write(f"  class_weight:      {r0.get('class_weight')}\n")
        f.write(f"  gate_feat:         {r0.get('gate_feat')}\n")
        f.write(f"  gate_teacher:      {r0.get('gate_teacher')}\n")
        f.write(f"  gate_hints:        {r0.get('gate_hints')}\n")
        f.write(f"  select:            {r0.get('select')}\n")
        f.write(f"  mask_randomize:    {r0.get('mask_randomize')}\n")
        f.write(f"  total_electrodes:  {total}\n")
        f.write(f"  fixed_electrodes:  {num_fixed} ({ratio*100:.1f}% of total)\n")
        f.write(f"  variable_target:   {total - num_fixed} "
                f"(可変チャネル {fold_results[targets[0]].get('n_variable_channels')} 本から選ぶ)\n")
        f.write(f"  full_fix_mode:     {fold_results[targets[0]].get('full_fix_mode')}\n\n")
        for title, summ in [('全ターゲット', summary_all),
                            ('両クラスを持つ被験者のみ', summary_balanced)]:
            if summ is None:
                continue
            f.write(f"----- {title} (n={summ['n_subjects']}) -----\n")
            for key, label in REPORT_METRICS:
                v = summ[key]
                if v is None:
                    continue
                pct = key in PCT_METRICS
                unit, fmt = ('%', '.2f') if pct else ('', '.4f')
                f.write(f"  {label}: {v['mean']:{fmt}}{unit} ± {v['std']:{fmt}}{unit}\n")
            sel = summ['avg_selected_electrodes']
            f.write(f"  Electrodes: {sel['mean']:.2f} ± {sel['std']:.2f} ch (目標 {total}ch)\n\n")
        f.write("Per subject:\n")
        for s in targets:
            r = fold_results[s]
            mark = '' if s in balanced else '  (degenerate)'
            f.write(f"  Sub{s}: n={r['n_test']} (awake={r['n_class0_true']}/"
                    f"fatigue={r['n_class1_true']}) acc={r['acc']:.2f}% "
                    f"prec={r['macro_precision']:.4f} rec={r['macro_recall']:.4f} "
                    f"f1={r['macro_f1']:.4f} sel={r['avg_selected_electrodes']:.2f}ch "
                    f"fixed={r['fixed_channels']}{mark}\n")
        f.write("\n===== Pooled Classification Report =====\n")
        f.write(report + "\n")
        f.write("===== Electrode Selection Rate (overall) =====\n")
        f.write("(fixed=このfold数の割合で固定電極に選ばれた)\n")
        for rank, i in enumerate(np.argsort(overall_usage)[::-1], 1):
            f.write(f"{rank:2d}. {CHANNEL_NAMES[i]:4s} {overall_usage[i]:.4f} "
                    f"(fixed {fixed_rate[i]*100:5.1f}%)\n")
        f.write("\n--- Per Class ---\n")
        for c in range(NUM_CLASSES):
            f.write(f"[{CLASS_LABELS[c]}]\n")
            for rank, i in enumerate(np.argsort(class_distribution[c])[::-1], 1):
                f.write(f"  {rank:2d}. {CHANNEL_NAMES[i]:4s} {class_distribution[c][i]:.4f}\n")
            f.write("\n")

    out = {
        'n_total_electrodes': total,
        'n_fixed_electrodes': num_fixed,
        'fixed_ratio': ratio,
        'n_variable_target': total - num_fixed,
        'n_folds': len(targets),
        'targets': targets,
        'degenerate_targets': degenerate,
        'summary_all_targets': summary_all,
        'summary_balanced_targets': summary_balanced,
        'pooled': pooled,
        'electrode_selection_rate': {CHANNEL_NAMES[i]: float(overall_usage[i])
                                     for i in range(NUM_CHANNELS)},
        'fixed_rate_across_folds': {CHANNEL_NAMES[i]: float(fixed_rate[i])
                                    for i in range(NUM_CHANNELS)},
        'fixed_channels_per_fold': {str(s): fold_results[s]['fixed_channels']
                                    for s in targets},
        'per_subject': {str(s): fold_results[s] for s in targets},
    }
    with open(os.path.join(fix_dir, 'loso_results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n===== Total {total}ch / Fixed {num_fixed}ch "
              f"({ratio*100:.0f}% fixed, {len(targets)} folds) =====")
        for title, summ in [('全ターゲット', summary_all),
                            ('両クラスを持つ被験者のみ', summary_balanced)]:
            if summ is None:
                continue
            print(f" {title} (n={summ['n_subjects']}名)")
            for key, label in REPORT_METRICS:
                v = summ[key]
                if v is None:
                    continue
                pct = key in PCT_METRICS
                unit, fmt = ('%', '.2f') if pct else ('', '.4f')
                print(f"   {label} : {v['mean']:{fmt}}{unit} ± {v['std']:{fmt}}{unit}")
            print(f"   Electrodes: {summ['avg_selected_electrodes']['mean']:.2f}ch "
                  f"(目標 {total}ch)")
    return out


def aggregate_all(total, base_dir, manifest, fix_counts):
    total_dir = os.path.join(base_dir, f'total{total}')
    os.makedirs(total_dir, exist_ok=True)

    rows, details = [], []
    for fx in fix_counts:
        r = aggregate_fix(total, fx, base_dir, manifest, verbose=True)
        if r is None:
            continue
        details.append(r)
        sa, sb = r['summary_all_targets'], r['summary_balanced_targets']

        def g(summ, key, stat='mean'):
            if summ is None or summ.get(key) is None:
                return float('nan')
            return round(float(summ[key][stat]), 6)

        rows.append({
            'n_total_electrodes': total,
            'n_fixed_electrodes': fx,
            'fixed_ratio': round(r['fixed_ratio'], 4),
            'n_variable_target': r['n_variable_target'],
            'n_folds': r['n_folds'],
            'avg_selected_electrodes': g(sa, 'avg_selected_electrodes'),
            'all_acc_mean': g(sa, 'acc'), 'all_acc_std': g(sa, 'acc', 'std'),
            'all_precision_mean': g(sa, 'macro_precision'),
            'all_recall_mean': g(sa, 'macro_recall'),
            'all_f1_mean': g(sa, 'macro_f1'), 'all_f1_std': g(sa, 'macro_f1', 'std'),
            'bal_acc_mean': g(sb, 'acc'), 'bal_acc_std': g(sb, 'acc', 'std'),
            'bal_precision_mean': g(sb, 'macro_precision'),
            'bal_recall_mean': g(sb, 'macro_recall'),
            'bal_f1_mean': g(sb, 'macro_f1'), 'bal_f1_std': g(sb, 'macro_f1', 'std'),
            'pooled_acc': round(r['pooled']['accuracy'] * 100.0, 4),
            'pooled_f1': round(r['pooled']['macro_f1'], 6),
        })

    if not rows:
        print(f"集約対象がありません: {total_dir}")
        return

    csv_path = os.path.join(total_dir, 'summary_fix_ratio.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {csv_path}")

    print(f"\n===== Summary (total {total}ch, sorted by number of fixed electrodes) =====")
    print(f"{'Fix':>3} | {'ratio':>5} | {'sel':>5} | {'balAcc':>7} | {'balF1':>7} | "
          f"{'allAcc':>7} | {'allF1':>7} | {'poolAcc':>7} | folds")
    print("-" * 82)
    for r in rows:
        print(f"{r['n_fixed_electrodes']:>3} | {r['fixed_ratio']:>5.2f} | "
              f"{r['avg_selected_electrodes']:>5.2f} | "
              f"{r['bal_acc_mean']:>7.2f} | {r['bal_f1_mean']:>7.4f} | "
              f"{r['all_acc_mean']:>7.2f} | {r['all_f1_mean']:>7.4f} | "
              f"{r['pooled_acc']:>7.2f} | {r['n_folds']}")

    valid = [r for r in rows if not np.isnan(r['bal_f1_mean'])]
    if valid:
        best = max(valid, key=lambda x: x['bal_f1_mean'])
        print(f"\n>>> Best by balanced-target macro F1: fixed {best['n_fixed_electrodes']}ch "
              f"/ dynamic {best['n_variable_target']}ch (ratio={best['fixed_ratio']:.2f}, "
              f"F1={best['bal_f1_mean']:.4f} ± {best['bal_f1_std']:.4f}, "
              f"Acc={best['bal_acc_mean']:.2f}%)")

    ns = [r['n_fixed_electrodes'] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, marker in [('bal_acc_mean', 'Accuracy (balanced targets)', 'o'),
                               ('all_acc_mean', 'Accuracy (all targets)', 'd')]:
        ax.plot(ns, [r[key] / 100.0 for r in rows], marker=marker, label=label)
    for key, label, marker in [('bal_f1_mean', 'Macro F1 (balanced targets)', 's'),
                               ('all_f1_mean', 'Macro F1 (all targets)', '^'),
                               ('bal_precision_mean', 'Macro Precision (balanced)', 'v'),
                               ('bal_recall_mean', 'Macro Recall (balanced)', '<')]:
        ax.plot(ns, [r[key] for r in rows], marker=marker, label=label)
    ax.set_xlabel(f'Number of FIXED electrodes (out of total {total})')
    ax.set_ylabel('Score')
    ax.set_title(f'LOSO metrics vs. fixed/dynamic ratio '
                 f'(paper_2class, fatigue 2-class, total {total}ch)')
    ax.set_xticks(ns)
    ax.grid(True); ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = os.path.join(total_dir, 'summary_metrics_vs_fixed.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    with open(os.path.join(total_dir, 'summary_fix_ratio.json'), 'w') as f:
        json.dump({'data_dir': DATA_DIR,
                   'n_total_electrodes': total,
                   'labeling_rule': manifest['labeling_rule'],
                   'excluded_subjects': manifest['excluded_subjects'],
                   'kept_subjects': manifest['kept_subjects'],
                   'balanced_targets': manifest['loso_target_subjects'],
                   'fix_counts': [d['n_fixed_electrodes'] for d in details],
                   'per_fix_count': details}, f, indent=2, ensure_ascii=False)
    print(f"Summary plot saved to: {plot_path}")


# =========================================================================
# 8. Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=DEFAULT_TOTAL_ELECTRODES,
                    help=f'総電極数 (1〜{NUM_CHANNELS})。既定 {DEFAULT_TOTAL_ELECTRODES} ')
    ap.add_argument('--epochs', type=int, default=FIXED_EPOCHS,
                    help=f'全fold共通の学習エポック数（既定 {FIXED_EPOCHS}）')
    ap.add_argument('--fix-counts', type=int, nargs='+', default=None,
                    help='探索する固定電極数（既定 0〜--total）')
    ap.add_argument('--only-fix', type=int, default=None,
                    help='この固定電極数だけ実行する')
    ap.add_argument('--only-target', type=int, default=None,
                    help='この被験者1名のfoldだけ実行する（並列実行用）。集約は行わない。')
    ap.add_argument('--targets', choices=['all', 'balanced'], default='all',
                    help="all: 採用15名全員をターゲットにする(既定) / "
                         "balanced: 少数クラス2%%以上の11名のみ")
    ap.add_argument('--limit-folds', type=int, default=None,
                    help='先頭N名だけ回す（動作確認用）')
    ap.add_argument('--teacher-importance', choices=['col', 'cls'], default='col',
                    help='Gateに渡す教師の電極重要度。col=electrode_attentionの列平均(既定) / '
                         'cls=CLS行attention(比較用)')
    ap.add_argument('--class-weight', choices=['none', 'balanced'],
                    default=CLASS_WEIGHT,
                    help=f'CrossEntropyLoss のクラス重み（既定 {CLASS_WEIGHT}）')
    ap.add_argument('--gate-feat', choices=['band', 'mean'], default=GATE_FEAT,
                    help=f'Gate に渡す EEG 特徴（既定 {GATE_FEAT}）')
    ap.add_argument('--gate-teacher', choices=['fold_const', 'per_sample', 'none'],
                    default=GATE_TEACHER,
                    help=f'教師の電極重要度の入れ方（既定 {GATE_TEACHER}）')
    ap.add_argument('--gate-hints', action='store_true', default=GATE_HINTS,
                    help='教師の予測確率を Gate に入れる（旧挙動）。既定は入れない')
    ap.add_argument('--select', choices=['topk', 'threshold'], default=SELECT,
                    help=f'電極の選び方（既定 {SELECT}）')
    ap.add_argument('--mask-randomize', type=float, default=MASK_RANDOMIZE,
                    help=f'学習時にこの確率で可変チャネルのマスクをランダムなk本に差し替える（既定 {MASK_RANDOMIZE}）')
    ap.add_argument('--full-fix', choices=['static', 'gate'], default='static',
                    help='固定数が総電極数と等しいとき(F=TOTAL)の扱い。')
    ap.add_argument('--ranking', choices=['fold', 'global'], default='fold',
                    help='固定電極の選定順。fold=そのfoldの学習被験者だけから作る(既定/リークなし) / '
                         'global=全fold共通ランキング(比較用)')
    ap.add_argument('--teacher-cache-dir', type=str, required=True,
                    help='【必須】生成済みの教師出力キャッシュが保存されているディレクトリのパスを指定してください')
    ap.add_argument('--jobs', type=int, default=None,
                    help='同時実行プロセス数（既定 GPU数×%d）' % PROCESSES_PER_GPU)
    ap.add_argument('--aggregate-only', action='store_true',
                    help='学習せず、保存済みのfold結果から集約だけ行う')
    ap.add_argument('--save-path', type=str, default=RESEARCH_BASE_DIR)
    args = ap.parse_args()

    if not (1 <= args.total <= NUM_CHANNELS):
        ap.error(f"--total は 1〜{NUM_CHANNELS} を指定してください (指定値: {args.total})")

    base_dir = args.save_path
    os.makedirs(base_dir, exist_ok=True)

    manifest = json.load(open(os.path.join(DATA_DIR, 'paper_manifest.json')))
    kept = manifest['kept_subjects']              # 採用15名
    balanced = manifest['loso_target_subjects']   # 両クラスを十分持つ11名

    fix_counts = args.fix_counts if args.fix_counts else list(range(0, args.total + 1))
    if args.only_fix is not None:
        fix_counts = [args.only_fix]
    fix_counts = [fx for fx in fix_counts if 0 <= fx <= args.total]
    if not fix_counts:
        ap.error(f"固定電極数は 0〜{args.total} の範囲で指定してください")

    if args.aggregate_only:
        aggregate_all(args.total, base_dir, manifest, fix_counts)
        return

    if args.only_target is not None:
        if args.only_target not in kept:
            raise SystemExit(f"--only-target {args.only_target} は採用被験者に含まれていません。候補: {kept}")
        targets = [args.only_target]
    else:
        targets = kept if args.targets == 'all' else balanced
        if args.limit_folds:
            targets = targets[:args.limit_folds]

    degenerate = [s for s in targets if s not in balanced]
    num_gpus = torch.cuda.device_count()

    print(f"データ    : {DATA_DIR}")
    print(f"ラベル    : {manifest['labeling_rule']}")
    print(f"除外      : {manifest['excluded_subjects']}")
    print(f"学習可    : {len(kept)}名 {kept}")
    print(f"ターゲット: {len(targets)}名 {targets}")
    if degenerate:
        print(f"  ※ うち{len(degenerate)}名 {degenerate} は少数クラスが極端に少なく、"
              f"macro Precision/Recall/F1が退化する。集計は両方を出す")
    print(f"総電極数  : {args.total}")
    print(f"固定電極数: {fix_counts} (残り {args.total}-F 本を動的選定)")
    print(f"エポック  : {args.epochs} (全fold固定 / validationもtestによる選択も無し)")
    print(f"教師重要度: {args.teacher_importance}")
    print(f"標準化    : Source 1方式 (Trainデータの平均と標準偏差を用いたグローバル標準化)")
    print(f"クラス重み: {args.class_weight}")
    print(f"Gate      : feat={args.gate_feat} / teacher={args.gate_teacher} "
          f"/ hints={args.gate_hints} / select={args.select} "
          f"/ mask_randomize={args.mask_randomize}")
    print(f"F=TOTAL時 : {args.full_fix} "
          f"({'Gateなし・上位TOTAL本のみ' if args.full_fix == 'static' else 'Gateあり・目標0本'})")
    print(f"固定順    : {args.ranking}"
          + ("  (foldの学習被験者だけから作成)" if args.ranking == 'fold'
             else f"  {GLOBAL_IMPORTANCE_RANKING}"))

    cache_dir = args.teacher_cache_dir
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"指定されたキャッシュディレクトリが見つかりません: {cache_dir}")
    print(f"\n指定された教師出力キャッシュを利用: {cache_dir}", flush=True)

    # 必要なキャッシュファイルがすべて揃っているか事前に確認
    for t in targets:
        p = teacher_cache_path(cache_dir, args.teacher_importance, t)
        if not os.path.exists(p):
            raise FileNotFoundError(f"キャッシュファイルが見つかりません: {p}\n"
                                    "正しいディレクトリを指定しているか確認してください。")
        print(f"  target{t}: {os.path.basename(p)} (確認済)", flush=True)

    if args.ranking == 'fold':
        print("\nfoldごとの固定電極 (重要度上位):", flush=True)
        for t in targets:
            cache = np.load(teacher_cache_path(cache_dir, args.teacher_importance, t))
            rk, _ = fold_ranking(cache, [s for s in kept if s != t])
            print(f"  target{t}: {rk[:args.total]}", flush=True)

    jobs = [(fx, t) for fx in fix_counts for t in targets]

    if args.jobs is not None:
        num_workers = max(1, args.jobs)
    elif num_gpus > 0:
        num_workers = max(1, num_gpus * PROCESSES_PER_GPU)
    else:
        num_workers = 1
    num_workers = min(num_workers, len(jobs))
    print(f"\n検出GPU   : {num_gpus} -> {num_workers} 並列 / 全 {len(jobs)} fold\n", flush=True)

    args_list = [(i, args.total, fx, t, kept, args.epochs, args.teacher_importance,
                  args.ranking, args.full_fix, num_gpus, base_dir,
                  args.class_weight, args.gate_feat, args.gate_teacher, args.gate_hints,
                  args.select, args.mask_randomize, cache_dir)
                 for i, (fx, t) in enumerate(jobs)]

    if num_workers == 1:
        for a in args_list:
            worker(*a)
    else:
        with mp.Pool(processes=num_workers) as pool:
            pool.starmap(worker, args_list, chunksize=1)

    if args.only_target is None:
        aggregate_all(args.total, base_dir, manifest, fix_counts)
    else:
        print(f"fold (target={args.only_target}) 完了。"
              f"全fold終了後に --aggregate-only で集約してください。")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()