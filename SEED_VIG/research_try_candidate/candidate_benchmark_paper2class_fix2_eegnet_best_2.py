"""
SEED-VIG データを用いた「どの電極を固定するか」の比較実験 (3本固定版, EEG 駆動のゲート)。

【ベースライン版 ＋ グローバル標準化】
動的な電極選択（Gateネットワーク）や探索の枠組みはそのまま維持し、
平滑化（LDSや移動平均）などの追加処理をすべて撤廃した純粋なコードです。

※データリークを防ぐため、Trainデータ全体の平均と標準偏差を用いた
グローバル標準化 (Global Normalization) を組み込んでいます。
※教師キャッシュは最適電極数探索 (loso_researchbestnum) で作成したものを再利用します。
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import traceback

# プロジェクトルートを import パスに追加 (select_channel / model を解決するため)
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
from scipy import stats
from scipy.interpolate import griddata
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, TensorDataset

from select_channel.select_net_class import gcn_select_net
from model.EEGNet import CustomEEGNet

warnings.filterwarnings("ignore")

# =========================================================================
# 設定
# =========================================================================
DATA_DIR = '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/processedData/subject_wise_2class'

# ベースライン (グローバル標準化のみ) で学習した教師モデルのパス
CHEAT_SHEET_DIR = os.environ.get(
    'CHEAT_SHEET_DIR',
    '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/'
    'LOSO_pretrainedmodel/result_baseline/baseline_global_norm')

CHEAT_SHEET_TAG = '_'.join([p for p in CHEAT_SHEET_DIR.rstrip('/').split('/')[-2:] if p])

RESEARCH_BASE_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/'
                     'research_fix_electrode/'
                     'results_candidates_seedvig_2class_baseline_fix2_best_2/')

NUM_CLASSES = 2
CLASS_NAMES = ['Awake', 'Fatigue']
CLASS_LABELS = {0: 'Awake', 1: 'Fatigue'}
CLASS_ORDER = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]

# SEED-VIG の 'chn' の並びから CPZ を除いた17ch
CHANNEL_NAMES = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
                 'CP1', 'CP2', 'P1', 'PZ', 'P2',
                 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']
NUM_CHANNELS = len(CHANNEL_NAMES)
NUM_BANDS = 5
CH_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}

# LOSO の対象となる全セッション
SUBJECT_LIST = [
    '1_20151124_noon_2', '2_20151106_noon', '3_20151024_noon', '4_20151105_noon',
    '4_20151107_noon', '5_20141108_noon', '5_20151012_night', '6_20151121_noon',
    '7_20151015_night', '8_20151022_noon', '9_20151017_night', '10_20151125_noon',
    '11_20151024_night', '12_20150928_noon', '13_20150929_noon', '14_20151014_night',
    '15_20151126_night', '16_20151128_night', '17_20150925_noon', '18_20150926_noon',
    '19_20151114_noon', '20_20151129_night', '21_20151016_noon'
]
SUBJECT_INDEX = {s: i for i, s in enumerate(SUBJECT_LIST)}
MIN_MINORITY_RATIO = 0.02

# 電極数設定: 最適電極数探索での最良点(14ch)をデフォルトとし、固定数を3に設定
DEFAULT_TOTAL_ELECTRODES = 10
DEFAULT_FIXED_ELECTRODES = 2

GLOBAL_IMPORTANCE_RANKING = ['CP1', 'FT7', 'CP2', 'P2', 'T8', 'T7', 'FT8', 'TP8', 'PZ', 'O2', 'OZ', 'PO4', 'P1', 'O1', 'PO3', 'TP7', 'POZ']

SEED = 42
BATCH_SIZE = 128
LR_GATE = 3e-3
LR_CLASSIFIER = 1e-4
WEIGHT_DECAY = 0.03
LAMBDA_SPARSITY = 1e-3
FIXED_EPOCHS = 100

# --- 前処理 ---
NORMALIZE = 'global'
CLASS_WEIGHT = 'none'

# --- Gate の設計 ---
GATE_FEAT = 'band'
GATE_TEACHER = 'fold_const'
GATE_HINTS = False
SELECT = 'topk'
MASK_RANDOMIZE = 0.3

PROCESSES_PER_GPU = 2
TEACHER_CACHE_DIRNAME = 'teacher_cache'
MAX_INTERP_DIST = 0.40

# 教師出力キャッシュの共有元
SHARED_CACHE_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/'
                    'research_number_of_electrode/'
                    'results_bestnum_seedvig_2class_baseline/teacher_cache')

# 報告する評価指標
# ★変更点1: Metricsの並び順と対象をBalancedに変更
METRICS = [('acc', 'Accuracy(%)'),
           ('macro_f1', 'Macro F1'),
           ('kappa', "Cohen's Kappa")]
METRIC_KEYS = [k for k, _ in METRICS]
METRIC_LABEL = dict(METRICS)
PCT_METRICS = {'acc'}

REPORT_METRICS = [('acc', 'Accuracy '),
                  ('macro_precision', 'Precision'),
                  ('macro_recall', 'Recall   '),
                  ('macro_f1', 'F1-Score '),
                  ('kappa', 'Kappa    ')]

COORDS = {
    'FT7': (-0.82, 0.42), 'FT8': (0.82, 0.42),
    'T7': (-0.95, 0.0), 'T8': (0.95, 0.0),
    'TP7': (-0.85, -0.35), 'TP8': (0.85, -0.35),
    'CP1': (-0.25, -0.30), 'CP2': (0.25, -0.30),
    'P1': (-0.26, -0.58), 'PZ': (0.0, -0.58), 'P2': (0.26, -0.58),
    'PO3': (-0.32, -0.78), 'POZ': (0.0, -0.78), 'PO4': (0.32, -0.78),
    'O1': (-0.27, -0.93), 'OZ': (0.0, -0.95), 'O2': (0.27, -0.93),
}

# =========================================================================
# 比較候補 (3本固定)
# =========================================================================
CANDIDATES = {
    'proposed': dict(
        channels=None,
        ranking='global',
        group='proposed',
        rationale='提案手法の変種。全fold共通の重要度ランキング上位3本を固定する。'
                  'fold別ランキングとの差分から、被験者集合ごとに作り直す効果を切り分ける。'),

    'occipital_alpha': dict(
        channels=['O1', 'O2'],
        ranking=None,
        group='literature',
        rationale='眠気の進行に伴うα帯域パワー増大を捉える後頭3本 (O1/O2/OZ)。左右対称。'),

    'parietal_attention': dict(
        channels=['CP1', 'CP2'],
        ranking=None,
        group='literature',
        rationale='覚醒度低下に伴う注意資源の減衰を捉える中心-頭頂3本 (CP1/CP2/PZ)。左右対称。'),

    'temporal_lateral': dict(
        channels=['T7', 'T8'],
        ranking=None,
        group='literature',
        rationale='側頭鎖。正中電極がないため、左右ペア (T7/T8) と左後側頭 (TP7) の非対称な3本構成。'),

    'anterior_posterior': dict(
        channels=['FT7', 'FT8'],
        ranking=None,
        group='literature',
        rationale='前方θと後頭αを同時に取る構成。前方の代替 (FT7/FT8) と後頭正中 (OZ) の3本。左右対称。'),

    'temporal_occipital': dict(
        channels=['TP7', 'TP8'],
        ranking=None,
        group='literature',
        rationale='後側頭 (TP7/TP8) と後頭 (OZ) の組み合わせ。左右対称。'),

    'uniform_coverage': dict(
        channels=['FT7', 'CP2'],
        ranking=None,
        group='literature',
        rationale='前側頭、中心頭頂、後頭から1本ずつ散らした均等空間サンプリング。'),

    'left_hemisphere': dict(
        channels=['FT7', 'TP7'],
        ranking=None,
        group='control',
        rationale='左半球のみ3本 (対照)。側方性が一方に偏った場合の影響を見る。'),

    'right_hemisphere': dict(
        channels=['FT8', 'TP8'],
        ranking=None,
        group='control',
        rationale='右半球のみ3本 (対照)。左半球の完全な鏡像。'),

    'frontal_control': dict(
        channels=['FT7', 'FT8'],
        ranking=None,
        group='control',
        rationale='SEED-VIGで最も前方に近い電極 (FT7/FT8) と正中 (PZ) の組み合わせ。ネガティブコントロール。左右対称。'),
}

GROUP_COLORS = {'proposed': '#c0504d', 'literature': '#3b6fb6', 'control': '#9aa4b1', 'random': '#7fa66b'}

# =========================================================================
# データ処理 & ユーティリティ
# =========================================================================
def build_manifest():
    kept, balanced, missing, counts = [], [], [], {}
    for s in SUBJECT_LIST:
        label_path = os.path.join(DATA_DIR, f'label_{s}.npy')
        eeg_path = os.path.join(DATA_DIR, f'eeg_{s}.npy')
        if not (os.path.exists(label_path) and os.path.exists(eeg_path)):
            missing.append(s)
            continue
        y = np.load(label_path)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        counts[s] = {'n': int(len(y)), 'n_awake': n0, 'n_fatigue': n1}
        kept.append(s)
        if min(n0, n1) >= MIN_MINORITY_RATIO * len(y):
            balanced.append(s)

    return {
        'kept_subjects': kept,
        'loso_target_subjects': balanced,
        'labeling_rule': 'PERCLOS < 0.35 -> 0 (Awake), >= 0.35 -> 1 (Fatigue)',
        'excluded_subjects': missing,
        'class_counts': counts,
    }


def normalize_subject(x, mode):
    if mode in ('none', 'global'):
        return x
    if mode == 'subject':
        return (x - x.mean()) / (x.std() + 1e-8)
    if mode == 'channel_band':
        mu = x.mean(axis=(0, 1), keepdims=True)
        sig = x.std(axis=(0, 1), keepdims=True)
        return (x - mu) / (sig + 1e-8)
    raise ValueError(f'unknown normalize mode: {mode}')


def prep_tag(normalize):
    return f'{CHEAT_SHEET_TAG}_{normalize}'


def load_subject_x(subject_id, normalize):
    x = np.load(os.path.join(DATA_DIR, f'eeg_{subject_id}.npy')).astype(np.float32)
    return normalize_subject(x, normalize)


def load_subjects(subject_ids, normalize):
    xs, ys = [], []
    for s in subject_ids:
        xs.append(load_subject_x(s, normalize))
        ys.append(np.load(os.path.join(DATA_DIR, f'label_{s}.npy')))
    x = torch.FloatTensor(np.concatenate(xs, axis=0))
    y = torch.LongTensor(np.concatenate(ys, axis=0).astype(np.int64))
    return x, y


def class_weights(subject_ids, mode, device):
    if mode == 'none':
        return None
    y = np.concatenate([np.load(os.path.join(DATA_DIR, f'label_{s}.npy'))
                        for s in subject_ids])
    n = len(y)
    counts = np.array([(y == c).sum() for c in range(NUM_CLASSES)], dtype=np.float64)
    if mode == 'balanced':
        w = np.where(counts > 0, n / (NUM_CLASSES * np.maximum(counts, 1)), 1.0)
        return torch.FloatTensor(w).to(device)
    raise ValueError(f'unknown class_weight mode: {mode}')


def fold_constant_importance(teacher_cache, train_subjects):
    return np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects], axis=0).mean(axis=0)


def fold_ranking(teacher_cache, train_subjects):
    imp = np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects], axis=0).mean(axis=0)                      
    order = np.argsort(imp)[::-1]
    return [CHANNEL_NAMES[i] for i in order], imp


def resolve_ranking(mode, teacher_cache, train_subjects):
    if mode == 'global':
        return list(GLOBAL_IMPORTANCE_RANKING), None
    return fold_ranking(teacher_cache, train_subjects)


def make_loader(subject_ids, shuffle, teacher_cache, normalize,
                imp_override=None, generator=None, train_mean=None, train_std=None):
    x, y = load_subjects(subject_ids, normalize)
    
    if normalize == 'global' and train_mean is not None and train_std is not None:
        x = (x - train_mean) / (train_std + 1e-8)
        
    hints = torch.FloatTensor(
        np.concatenate([teacher_cache[f'hints_{s}'] for s in subject_ids], axis=0))
    if imp_override is None:
        imp = torch.FloatTensor(
            np.concatenate([teacher_cache[f'imp_{s}'] for s in subject_ids], axis=0))
    else:
        imp = torch.FloatTensor(
            np.repeat(np.asarray(imp_override, dtype=np.float32)[None, :], len(x), axis=0))
    
    ds = TensorDataset(x, y, hints, imp)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


def find_cheat_sheet(target):
    candidates = [os.path.join(CHEAT_SHEET_DIR, f'model_target_{target}.pth'),
                  os.path.join(CHEAT_SHEET_DIR, f'best_model_target_{target}.pth')]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def teacher_cache_path(cache_dir, mode, target, normalize):
    return os.path.join(cache_dir, f'teacher_{mode}_{prep_tag(normalize)}_target_{target}.npz')

def cls_row_importance(attention_weights):
    rows = []
    for layer in attention_weights:
        a = torch.stack(layer)                
        rows.append(a.mean(dim=0)[:, 0, 1:])  
    return torch.stack(rows).mean(dim=0)      

@torch.no_grad()
def teacher_outputs(cheat_sheet_model, x, mode):
    teacher_logits, attn_w, electrode_attention, _ = cheat_sheet_model(x)
    teacher_hints = torch.softmax(teacher_logits, dim=1)

    if mode == 'cls':
        teacher_imp = cls_row_importance(attn_w)                 
    else:
        teacher_imp = electrode_attention.mean(dim=0)[:, 1:]

    mean_imp = teacher_imp.mean(dim=1, keepdim=True)
    std_imp = teacher_imp.std(dim=1, keepdim=True)
    teacher_importance = (teacher_imp - mean_imp) / (std_imp + 1e-6)
    return teacher_hints, teacher_importance

@torch.no_grad()
def build_teacher_cache(target, subjects, mode, device, cache_dir, normalize):
    path = teacher_cache_path(cache_dir, mode, target, normalize)
    if os.path.exists(path):
        return path

    cheat_path = find_cheat_sheet(target)
    if cheat_path is None:
        raise FileNotFoundError(f"教師モデルが見つかりません: {CHEAT_SHEET_DIR}")

    model = gcn_select_net(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(cheat_path, map_location=device))
    model.eval()   

    train_mean, train_std = None, None
    if normalize == 'global':
        train_subjects = [s for s in subjects if s != target]
        x_train, _ = load_subjects(train_subjects, 'none')
        train_mean = x_train.mean(dim=0, keepdim=True)
        train_std = x_train.std(dim=0, keepdim=True)
        del x_train

    def compute(s):
        x, _ = load_subjects([s], 'none' if normalize == 'global' else normalize)
        if normalize == 'global':
            x = (x - train_mean) / (train_std + 1e-8)
        hints, imps = [], []
        for i in range(0, len(x), BATCH_SIZE):
            h, m = teacher_outputs(model, x[i:i + BATCH_SIZE].to(device), mode)
            hints.append(h.cpu())
            imps.append(m.cpu())
        return torch.cat(hints).numpy(), torch.cat(imps).numpy()

    out = {}
    for s in subjects:
        out[f'hints_{s}'], out[f'imp_{s}'] = compute(s)

    tmp = path + '.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, path)   
    del model
    torch.cuda.empty_cache()
    return path

def ensure_teacher_cache(target, subjects, mode, device, cache_dir, normalize):
    dst = teacher_cache_path(cache_dir, mode, target, normalize)
    if not os.path.exists(dst):
        src_path = teacher_cache_path(SHARED_CACHE_DIR, mode, target, normalize)
        if os.path.exists(src_path) and os.path.abspath(src_path) != os.path.abspath(dst):
            os.makedirs(cache_dir, exist_ok=True)
            try:
                os.symlink(os.path.abspath(src_path), dst)
            except FileExistsError:
                pass
    return build_teacher_cache(target, subjects, mode, device, cache_dir, normalize)

def make_random_candidates(n, fixed_n, seed=SEED):
    rng = random.Random(seed)
    out = {}
    for k in range(1, n + 1):
        chs = sorted(rng.sample(CHANNEL_NAMES, fixed_n), key=CHANNEL_NAMES.index)
        out[f'random{k}'] = dict(
            channels=chs, ranking=None, group='random',
            rationale=f'ランダムに選んだ {fixed_n} 本 (対照 #{k}, seed={seed})。')
    return out

def resolve_candidates(fixed_n, n_random, ranking_mode):
    specs = dict(CANDIDATES)
    specs.update(make_random_candidates(n_random, fixed_n))

    out = {}
    for name, spec in specs.items():
        chs = spec['channels']
        if chs is None:
            if name == 'proposed_global' and ranking_mode == 'global':
                continue
            out[name] = dict(spec, channels=None, indices=None)
            continue
        unknown = [c for c in chs if c not in CH_INDEX]
        if unknown:
            raise ValueError(f"{name}: 未知のチャネル名 {unknown}")
        if len(chs) < fixed_n:
            print(f"  [skip] {name}: 電極数 {len(chs)} < --fixed {fixed_n}")
            continue
        chs = list(chs[:fixed_n])
        out[name] = dict(spec, channels=chs, indices=[CH_INDEX[c] for c in chs])
    return out

def fold_fixed_channels(name, spec, fixed_n, teacher_cache, train_subjects, ranking_mode):
    if spec['channels'] is not None:
        return list(spec['channels'])
    mode = spec.get('ranking') or ranking_mode
    if mode == 'global':
        return list(GLOBAL_IMPORTANCE_RANKING[:fixed_n])
    ranking, _ = fold_ranking(teacher_cache, train_subjects)
    return list(ranking[:fixed_n])

def candidate_dir(base_dir, total, fixed, name):
    return os.path.join(base_dir, f'total{total}_fix{fixed}', f'cand_{name}')


# =========================================================================
# Gate Mechanism / ConceptDynamicNet
# =========================================================================
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


class ConceptDynamicNet(nn.Module):
    def __init__(self, classifier_model, num_channels=17,
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
# 可視化処理
# =========================================================================
def plot_head_map(ax, channel_names, usage_weights, title=""):
    x_coords, y_coords, z_values = [], [], []
    for i, name in enumerate(channel_names):
        if name in COORDS:
            x, y = COORDS[name]
            x_coords.append(x)
            y_coords.append(y)
            z_values.append(usage_weights[i])

    elec_x, elec_y = np.array(x_coords), np.array(y_coords)
    x_coords, y_coords, z_values = np.array(x_coords), np.array(y_coords), np.array(z_values)

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
    grid_z = griddata((x_coords, y_coords), z_values, (grid_x, grid_y), method='cubic', fill_value=0.0)
    grid_z = np.clip(grid_z, 0.0, 1.0)

    dist = np.sqrt(grid_x ** 2 + grid_y ** 2)
    nearest = np.min(np.sqrt((grid_x[..., None] - elec_x) ** 2 + (grid_y[..., None] - elec_y) ** 2), axis=-1)
    grid_z = np.ma.masked_where((dist > 1.0) | (nearest > MAX_INTERP_DIST), grid_z)

    levels = np.linspace(0.0, 1.0, 100)
    im = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='jet', vmin=0.0, vmax=1.0)
    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, linewidth=1.5))
    ax.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k', linewidth=1.5)

    for i, name in enumerate(channel_names):
        if name in COORDS:
            cx, cy = COORDS[name]
            ax.scatter(cx, cy, c='k', s=18, marker='o', alpha=0.5, zorder=5)
            ax.text(cx, cy + 0.075, name, fontsize=7.5, ha='center', va='center', color='black', alpha=0.85, zorder=6)

    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.4)
    ax.set_aspect('equal')
    return im


def plot_class_topography(class_distribution, out_path, total, num_fixed, n_folds):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(6 * NUM_CLASSES, 5.5))
    fig.suptitle(f'Electrode Selection (Total {total}ch / Fixed {num_fixed}ch, mean of {n_folds} LOSO folds)', fontsize=15)
    im = None
    for c in range(NUM_CLASSES):
        im = plot_head_map(axes[c], CHANNEL_NAMES, class_distribution[c], title=CLASS_LABELS[c])
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
    plt.title(f'Electrode Selection Rate (Total {total}ch / Fixed {num_fixed}ch, {n_folds} LOSO folds)')
    plt.xlabel("Channel"); plt.ylabel("Selection Probability")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(trues, preds, out_path, title):
    cm = confusion_matrix(trues, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================================
# 学習・推論・ワーカー
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
        if x.size(0) == 0: continue
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


def run_fold(name, spec, target, kept, total, fixed, epochs, teacher_importance,
             ranking_mode, device, cand_dir, cache_dir,
             normalize=NORMALIZE, class_weight=CLASS_WEIGHT,
             gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER, gate_hints=GATE_HINTS,
             select=SELECT, mask_randomize=MASK_RANDOMIZE):
    train_subjects = [s for s in kept if s != target]

    seed = SEED + SUBJECT_INDEX[target]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)

    tag = f'[{name}/{target}]'

    # --- Trainデータ全体から mean / std を計算 (Global Normalization) ---
    train_mean, train_std = None, None
    if normalize == 'global':
        x_train_raw, _ = load_subjects(train_subjects, 'none')
        train_mean = x_train_raw.mean(dim=0, keepdim=True)
        train_std = x_train_raw.std(dim=0, keepdim=True)
        del x_train_raw

    # --- 教師キャッシュ ---
    cache_path = ensure_teacher_cache(target, kept, teacher_importance, device, cache_dir, normalize)
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

    fixed_names = fold_fixed_channels(name, spec, fixed, teacher_cache, train_subjects, ranking_mode)
    fixed_indices = [CH_INDEX[c] for c in fixed_names]
    variable_indices = [i for i in range(NUM_CHANNELS) if i not in fixed_indices]
    target_var_electrodes = max(0, total - fixed)

    print(f"{tag} train {len(train_subjects)}名 / test [{target}] | fixed={fixed_names} | var_target={target_var_electrodes}/{len(variable_indices)} | epochs={epochs} | normalize={normalize} | {device}", flush=True)

    classifier_model = CustomEEGNet(numclasses=NUM_CLASSES).to(device)
    model = ConceptDynamicNet(
        classifier_model, num_channels=NUM_CHANNELS, fixed_indices=fixed_indices,
        variable_indices=variable_indices, num_classes=NUM_CLASSES,
        n_select_var=target_var_electrodes, gate_feat=gate_feat, use_hints=gate_hints,
        imp_dim=imp_dim, select=select, mask_randomize=mask_randomize, n_bands=NUM_BANDS,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.gate.parameters(), 'lr': LR_GATE},
        {'params': model.classifier.parameters(), 'lr': LR_CLASSIFIER},
    ], weight_decay=WEIGHT_DECAY)
    
    weights = class_weights(train_subjects, class_weight, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_loader = make_loader(train_subjects, shuffle=True, teacher_cache=teacher_cache,
                               normalize=normalize, imp_override=imp_override, generator=gen,
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
        print(f"    {tag} Epoch {ep+1:3d}/{epochs} | loss {curve_train_loss[-1]:.4f} | acc {curve_train_acc[-1]:5.2f}% | usage {curve_usage[-1]:.1f}ch", flush=True)

    test_loader = make_loader([target], shuffle=False, teacher_cache=teacher_cache,
                              normalize=normalize, imp_override=imp_override,
                              train_mean=train_mean, train_std=train_std)
    tm, trues, preds, gate_sum, gate_counts = evaluate(model, test_loader, device, loss_fn)

    labels = list(range(NUM_CLASSES))
    per_rec = recall_score(trues, preds, labels=labels, average=None, zero_division=0)
    per_f1 = f1_score(trues, preds, labels=labels, average=None, zero_division=0)
    tm['kappa'] = float(cohen_kappa_score(trues, preds, labels=labels))
    if np.isnan(tm['kappa']): tm['kappa'] = 0.0
    for i, c in enumerate(CLASS_ORDER):
        tm[f'recall_{c}'] = float(per_rec[i])
        tm[f'f1_{c}'] = float(per_f1[i])

    print(f"{tag} test 覚醒{tm['n_class0_true']}/疲労{tm['n_class1_true']} -> Acc {tm['acc']:.2f}% | F1 {tm['macro_f1']:.4f} | 選定 {tm['avg_selected_electrodes']:.2f}ch", flush=True)

    result = {**tm, 'candidate': name, 'group': spec['group'], 'target': target,
              'n_total_electrodes': total, 'n_fixed_electrodes': fixed,
              'n_variable_target': target_var_electrodes,
              'n_variable_channels': len(variable_indices),
              'fixed_channels': fixed_names, 'ranking_mode': ranking_mode,
              'n_test': len(trues), 'train_subjects': train_subjects, 'epochs': epochs,
              'teacher_importance': teacher_importance,
              'teacher_dir': CHEAT_SHEET_DIR, 'normalize': normalize,
              'class_weight': class_weight, 'gate_feat': gate_feat, 'gate_teacher': gate_teacher,
              'gate_hints': gate_hints, 'select': select, 'mask_randomize': mask_randomize}

    with open(os.path.join(cand_dir, f'fold_target_{target}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(cand_dir, f'fold_target_{target}.npz'),
             trues=np.array(trues), preds=np.array(preds),
             gate_sum=gate_sum.numpy(), gate_counts=gate_counts.numpy(),
             fixed_indices=np.array(fixed_indices, dtype=np.int64),
             curve_train_loss=np.array(curve_train_loss),
             curve_train_acc=np.array(curve_train_acc),
             curve_usage=np.array(curve_usage))
    torch.save(model.state_dict(), os.path.join(cand_dir, f'concept_dynamic_net_target_{target}.pth'))

    del model, classifier_model, optimizer
    torch.cuda.empty_cache()
    return result


def worker(job_idx, name, spec, target, kept, total, fixed, epochs,
           teacher_importance, ranking_mode, num_gpus, base_dir, cache_dir,
           normalize=NORMALIZE, class_weight=CLASS_WEIGHT,
           gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER, gate_hints=GATE_HINTS,
           select=SELECT, mask_randomize=MASK_RANDOMIZE):
    gpu_id = job_idx % num_gpus if num_gpus > 0 else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    cand_dir = candidate_dir(base_dir, total, fixed, name)
    os.makedirs(cand_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        return run_fold(name, spec, target, kept, total, fixed, epochs,
                        teacher_importance, ranking_mode, device, cand_dir, cache_dir,
                        normalize, class_weight,
                        gate_feat, gate_teacher, gate_hints, select, mask_randomize)
    except Exception as e:
        print(f"[{name}/{target}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return None


# =========================================================================
# 集約処理
# =========================================================================
def summarize(fold_results, subset):
    sub = [fold_results[s] for s in subset if s in fold_results]
    if not sub: return None
    out = {'n_subjects': len(sub), 'subjects': list(subset)}
    for key, _ in REPORT_METRICS:
        v = np.array([r[key] for r in sub], dtype=float)
        v = v[~np.isnan(v)]
        out[key] = {'mean': float(v.mean()), 'std': float(v.std())} if len(v) else None
    for c in CLASS_ORDER:
        v = np.array([r[f'recall_{c}'] for r in sub], dtype=float)
        out[f'recall_{c}'] = {'mean': float(v.mean()), 'std': float(v.std())}
    sel = np.array([r['avg_selected_electrodes'] for r in sub], dtype=float)
    out['avg_selected_electrodes'] = {'mean': float(sel.mean()), 'std': float(sel.std())}
    return out


def aggregate_candidate(name, spec, total, fixed, base_dir, manifest, verbose=True):
    cand_dir = candidate_dir(base_dir, total, fixed, name)
    files = sorted(glob.glob(os.path.join(cand_dir, 'fold_target_*.json')))
    if not files: return None

    balanced = manifest['loso_target_subjects']
    fold_results, arrays = {}, {}
    for p in files:
        r = json.load(open(p))
        fold_results[r['target']] = r
        arrays[r['target']] = np.load(p.replace('.json', '.npz'))
        
    targets = sorted(fold_results, key=lambda s: SUBJECT_INDEX.get(s, len(SUBJECT_LIST)))
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
        'kappa': float(cohen_kappa_score(global_true, global_pred)),
    }
    report = classification_report(global_true, global_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)

    gate_sum = np.sum([arrays[t]['gate_sum'] for t in targets], axis=0)        
    gate_counts = np.sum([arrays[t]['gate_counts'] for t in targets], axis=0)  
    denom = gate_counts.copy()
    denom[denom == 0] = 1.0
    class_distribution = gate_sum / denom[:, None]
    overall_usage = gate_sum.sum(axis=0) / max(gate_counts.sum(), 1.0)

    fixed_rate = np.zeros(NUM_CHANNELS)
    for t in targets:
        fixed_rate[arrays[t]['fixed_indices'].astype(int)] += 1.0
    fixed_rate /= max(len(targets), 1)

    plot_confusion(global_true, global_pred, os.path.join(cand_dir, 'confusion_matrix_pooled.png'), f'{name} (total {total}ch / fixed {fixed}ch) pooled LOSO (Acc: {pooled["accuracy"]*100:.2f}%)')
    plot_class_topography(class_distribution, os.path.join(cand_dir, 'class_wise_topography.png'), total, fixed, len(targets))
    plot_selection_bar(overall_usage, fixed_rate, os.path.join(cand_dir, 'electrode_selection_rate.png'), total, fixed, len(targets))

    out = {
        'candidate': name, 'group': spec['group'], 'rationale': spec['rationale'],
        'n_total_electrodes': total, 'n_fixed_electrodes': fixed,
        'n_folds': len(targets), 'targets': targets, 'degenerate_targets': degenerate,
        'summary_all_targets': summary_all, 'summary_balanced_targets': summary_balanced,
        'pooled': pooled,
        'electrode_selection_rate': {CHANNEL_NAMES[i]: float(overall_usage[i]) for i in range(NUM_CHANNELS)},
        'fixed_rate_across_folds': {CHANNEL_NAMES[i]: float(fixed_rate[i]) for i in range(NUM_CHANNELS)},
        'fixed_channels_per_fold': {str(s): fold_results[s]['fixed_channels'] for s in targets},
        'per_subject': {str(s): fold_results[s] for s in targets},
    }
    with open(os.path.join(cand_dir, 'loso_results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out


def paired_table(per_fold, cands, subset, metric, ref='proposed'):
    rows = []
    if ref not in per_fold: return rows
    base = per_fold[ref]
    for name in cands:
        if name == ref or name not in per_fold: continue
        other = per_fold[name]
        common = [t for t in subset if t in base and t in other]
        if len(common) < 2:
            rows.append((name, float('nan'), float('nan'), len(common), float('nan'), float('nan')))
            continue
        a = np.array([other[t][metric] for t in common], dtype=float)
        b = np.array([base[t][metric] for t in common], dtype=float)
        d = a - b
        if np.allclose(d, 0): p_t = p_w = 1.0
        else:
            p_t = float(stats.ttest_rel(a, b).pvalue)
            try: p_w = float(stats.wilcoxon(a, b).pvalue)
            except Exception: p_w = float('nan')
        rows.append((name, float(d.mean()), float(d.std(ddof=1)), len(common), p_t, p_w))
    return rows


def aggregate_all(total, fixed, base_dir, manifest, cands):
    res_dir = os.path.join(base_dir, f'total{total}_fix{fixed}')
    os.makedirs(res_dir, exist_ok=True)

    balanced = manifest['loso_target_subjects']

    details, per_fold = [], {}
    for name, spec in cands.items():
        r = aggregate_candidate(name, spec, total, fixed, base_dir, manifest, verbose=True)
        if r is None: continue
        details.append(r)
        per_fold[name] = dict(r['per_subject'])   

    if not details:
        print(f"集約対象がありません: {res_dir}")
        return

    # ★変更点2: summary_candidates.json の出力を追加
    output_data = {
        'n_total_electrodes': total,
        'n_fixed_electrodes': fixed,
        'balanced_targets': balanced,
        'per_candidate': details
    }
    json_path = os.path.join(res_dir, 'summary_candidates.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to: {json_path}")

    rows = []
    for r in details:
        sa, sb = r['summary_all_targets'], r['summary_balanced_targets']

        def g(summ, key, stat='mean'):
            if summ is None or summ.get(key) is None: return float('nan')
            return round(float(summ[key][stat]), 6)

        row = {
            'candidate': r['candidate'], 'group': r['group'],
            'n_total_electrodes': total, 'n_fixed_electrodes': fixed, 'n_folds': r['n_folds'],
            'fixed_channels': ';'.join(sorted(set(c for v in r['fixed_channels_per_fold'].values() for c in v)) if r['candidate'].startswith('proposed') else next(iter(r['fixed_channels_per_fold'].values()))),
            'avg_selected_electrodes': g(sa, 'avg_selected_electrodes'),
            'all_acc_mean': g(sa, 'acc'), 'all_acc_std': g(sa, 'acc', 'std'),
            'all_precision_mean': g(sa, 'macro_precision'), 'all_recall_mean': g(sa, 'macro_recall'),
            'all_f1_mean': g(sa, 'macro_f1'), 'all_f1_std': g(sa, 'macro_f1', 'std'), 'all_kappa_mean': g(sa, 'kappa'),
            'bal_acc_mean': g(sb, 'acc'), 'bal_acc_std': g(sb, 'acc', 'std'),
            'bal_precision_mean': g(sb, 'macro_precision'), 'bal_recall_mean': g(sb, 'macro_recall'),
            'bal_f1_mean': g(sb, 'macro_f1'), 'bal_f1_std': g(sb, 'macro_f1', 'std'), 'bal_kappa_mean': g(sb, 'kappa'),
            'pooled_acc': round(r['pooled']['accuracy'] * 100.0, 4), 'pooled_f1': round(r['pooled']['macro_f1'], 6), 'pooled_kappa': round(r['pooled']['kappa'], 6),
        }
        for c in CLASS_ORDER:
            row[f'bal_recall_{c}'] = g(sb, f'recall_{c}')
        rows.append(row)

    rows.sort(key=lambda x: (-x['bal_f1_mean'] if not np.isnan(x['bal_f1_mean']) else 0))
    csv_path = os.path.join(res_dir, 'summary_candidates.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {csv_path}")

    # ===== テキストレポート =====
    lines = []
    add = lines.append
    add("===== Candidate benchmark: proposed vs. fatigue-motivated fixed electrode sets =====")
    add(f"protocol  : LOSO / total {total}ch のうち {fixed}ch を固定、残り {total-fixed}ch を Gate が動的選定")
    add("")

    hdr = (f"{'candidate':<22} {'group':<11}" + "".join(f"{lab:>17}" for _, lab in METRICS))
    add("--- Mean ± SD over LOSO folds (balanced targets) ---")
    add(hdr); add("-" * len(hdr))
    order = [r['candidate'] for r in rows]
    stat_of = {r['candidate']: r for r in details}
    for name in order:
        r = stat_of[name]
        sb = r['summary_balanced_targets']
        s = f"{name:<22} {r['group']:<11}"
        for key, _ in METRICS:
            v = sb.get(key) if sb else None
            s += "              n/a" if v is None else f"{v['mean']:>10.4f}±{v['std']:.3f}"
        add(s)
    add("")

    for subset_name, subset in [('balanced targets', balanced), ('all targets', details[0]['targets'])]:
        add(f"--- Paired comparison against 'proposed' ({subset_name}, same subjects) ---")
        for key, label in METRICS:
            add(f"  [{label}]")
            for name, dm, ds, n, p_t, p_w in paired_table(per_fold, order, subset, key):
                if np.isnan(dm): continue
                sig = '**' if min(p_t, p_w) < 0.01 else ('*' if min(p_t, p_w) < 0.05 else '')
                add(f"    {name:<22} delta={dm:+.4f} ±{ds:.4f} (n={n}, p_t={p_t:.3f}, p_w={p_w:.3f}){sig}")
            add("")

    txt = "\n".join(lines)
    with open(os.path.join(res_dir, 'candidate_summary.txt'), 'w') as f:
        f.write(txt + "\n")
    print("\n" + txt)

    # ===== Plot =====
    # ★変更点3: Plot処理をBalanced用に変更し、指標ごとに独立ソートするように調整
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.4 * len(METRICS), 5.0))
    for ax, (key, label) in zip(np.atleast_1d(axes), METRICS):
        # target を balanced 対象に変更 (stat_of[n]['summary_balanced_targets'])
        vals = [(n, (stat_of[n]['summary_balanced_targets'] or {}).get(key)) for n in order]
        vals = [(n, v) for n, v in vals if v is not None]
        # ★ 各パネル(指標)ごとに独立してソート (昇順: 上から高い値になるよう barh の特性に合わせる)
        vals.sort(key=lambda kv: kv[1]['mean'])
        
        names = [n for n, _ in vals]
        means = np.array([v['mean'] for _, v in vals])
        sds = np.array([v['std'] for _, v in vals])
        cols = [GROUP_COLORS.get(stat_of[n]['group'], '#888') for n in names]
        y = np.arange(len(names))
        ax.barh(y, means, xerr=sds, color=cols, alpha=.9, error_kw=dict(lw=1, capsize=2.5, ecolor='#444'))
        
        # foldごとのドットを描画 (balancedのみ)
        for yi, n in zip(y, names):
            pts = [per_fold[n][t][key] for t in balanced if t in per_fold[n] and key in per_fold[n][t]]
            if pts:
                ax.scatter(pts, np.full(len(pts), yi, dtype=float), s=8, color='black', alpha=.45, zorder=3)
        ax.set_yticks(y, names)
        
        floor = -1.0 if key == 'kappa' else 0.0
        if len(means) > 0:
            lo = max(floor, float(means.min() - 4 * (sds.max() + 1e-3)))
            ax.set_xlim(lo, float(means.max() + 2 * (sds.max() + 1e-3)))
        ax.set_xlabel(label); ax.set_title(label)
        
    fig.suptitle(f'Proposed vs. fatigue-motivated fixed electrode sets (total {total}ch / fixed {fixed}ch, balanced targets)', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, 'fig_candidates_metrics.png'))
    plt.close(fig)


# =========================================================================
# 6. Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser(
        description='提案パターン vs 疲労EEGの知見に基づく固定電極パターンの LOSO 比較 '
                    '(SEED-VIG Baseline)')
    ap.add_argument('--total', type=int, default=DEFAULT_TOTAL_ELECTRODES,
                    help=f'総電極数 (1〜{NUM_CHANNELS})。既定 {DEFAULT_TOTAL_ELECTRODES} ')
    ap.add_argument('--fixed', type=int, default=DEFAULT_FIXED_ELECTRODES,
                    help=f'固定電極数。既定 {DEFAULT_FIXED_ELECTRODES} ')
    ap.add_argument('--epochs', type=int, default=FIXED_EPOCHS,
                    help=f'全fold共通の学習エポック数（既定 {FIXED_EPOCHS}）')
    ap.add_argument('--only', nargs='+', default=None,
                    help='指定した候補名だけ実行する (スモークテスト用)')
    ap.add_argument('--only-candidate', type=str, default=None,
                    help='この候補だけ実行する (--only の単数版)')
    ap.add_argument('--only-target', type=str, default=None,
                    help='このセッション1件のfoldだけ実行する（並列実行用）。集約は行わない。')
    ap.add_argument('--targets', choices=['all', 'balanced'], default='all',
                    help="all: 読み込めた全セッションをターゲットにする(既定) / balanced: 少数クラスが2%%以上のセッションのみ")
    ap.add_argument('--limit-folds', type=int, default=None,
                    help='先頭N名だけ回す（動作確認用）')
    ap.add_argument('--random-controls', type=int, default=0,
                    help='ランダムな固定電極パターンを N 個追加する (対照)')
    ap.add_argument('--teacher-importance', choices=['col', 'cls'], default='col',
                    help='Gateに渡す教師の電極重要度。col=electrode_attentionの列平均(既定) / cls=CLS行attention(比較用)')
    ap.add_argument('--normalize', choices=['none', 'subject', 'channel_band', 'global'], default=NORMALIZE,
                    help=f'入力の標準化方式（既定 {NORMALIZE}）')
    ap.add_argument('--class-weight', choices=['none', 'balanced'], default=CLASS_WEIGHT,
                    help=f'CrossEntropyLoss のクラス重み（既定 {CLASS_WEIGHT}）')
    ap.add_argument('--gate-feat', choices=['band', 'mean'], default=GATE_FEAT,
                    help=f'Gate に渡す EEG 特徴（既定 {GATE_FEAT}）')
    ap.add_argument('--gate-teacher', choices=['fold_const', 'per_sample', 'none'], default=GATE_TEACHER,
                    help=f'教師の電極重要度の入れ方（既定 {GATE_TEACHER}）')
    ap.add_argument('--gate-hints', action='store_true', default=GATE_HINTS,
                    help='教師の予測確率を Gate に入れる。既定は入れない')
    ap.add_argument('--select', choices=['topk', 'threshold'], default=SELECT,
                    help=f'電極の選び方（既定 {SELECT}）')
    ap.add_argument('--mask-randomize', type=float, default=MASK_RANDOMIZE,
                    help=f'学習時に可変チャネルのマスクをランダムに差し替える確率（既定 {MASK_RANDOMIZE}）')
    ap.add_argument('--ranking', choices=['fold', 'global'], default='fold',
                    help="候補 'proposed' の固定電極の選定順。fold=そのfoldの学習被験者だけから作る(既定) / global=全fold共通ランキング")
    ap.add_argument('--jobs', type=int, default=None,
                    help='同時実行プロセス数（既定 GPU数×%d）' % PROCESSES_PER_GPU)
    ap.add_argument('--redo', action='store_true',
                    help='既に結果がある (候補, 被験者) も作り直す。既定はスキップして再開')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='学習せず、保存済みのfold結果から集約だけ行う')
    ap.add_argument('--save-path', type=str, default=RESEARCH_BASE_DIR)
    ap.add_argument('--cache-dir', type=str, default=None,
                    help='教師出力キャッシュの置き場')
    args = ap.parse_args()

    if not (1 <= args.total <= NUM_CHANNELS):
        ap.error(f"--total は 1〜{NUM_CHANNELS} を指定してください (指定値: {args.total})")
    if not (0 <= args.fixed <= args.total):
        ap.error(f"--fixed は 0〜{args.total} を指定してください (指定値: {args.fixed})")

    base_dir = args.save_path
    os.makedirs(base_dir, exist_ok=True)

    manifest = build_manifest()
    kept = manifest['kept_subjects']
    balanced = manifest['loso_target_subjects']
    if not kept:
        raise SystemExit(f"データが見つかりません: {DATA_DIR}")
    with open(os.path.join(base_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"総電極数  : {args.total} / 固定 {args.fixed} / 動的 {args.total - args.fixed}")
    
    cands = resolve_candidates(args.fixed, args.random_controls, args.ranking)
    only = list(args.only or [])
    if args.only_candidate: only.append(args.only_candidate)
    if only:
        cands = {k: v for k, v in cands.items() if k in only}
    if not cands:
        ap.error(f"--fixed {args.fixed} で実行できる候補がありません")
    for name, spec in cands.items():
        chs = ('重要度ランキング上位%d' % args.fixed if spec['channels'] is None else '; '.join(spec['channels']))
        print(f"  {name:<22} [{spec['group']:<11}] {chs}")

    if args.aggregate_only:
        aggregate_all(args.total, args.fixed, base_dir, manifest, cands)
        return

    if args.only_target is not None:
        targets = [args.only_target]
    else:
        targets = kept if args.targets == 'all' else balanced
        if args.limit_folds: targets = targets[:args.limit_folds]

    num_gpus = torch.cuda.device_count()

    # 教師出力キャッシュの作成・共有
    cache_dir = args.cache_dir or os.path.join(base_dir, TEACHER_CACHE_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    cache_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n教師出力キャッシュ (共有): {cache_dir}", flush=True)
    for t in targets:
        ensure_teacher_cache(t, kept, args.teacher_importance, cache_device, cache_dir, args.normalize)

    jobs = []
    skipped = 0
    for name in cands:
        for t in targets:
            done = os.path.join(candidate_dir(base_dir, args.total, args.fixed, name), f'fold_target_{t}.json')
            if not args.redo and os.path.exists(done):
                skipped += 1
                continue
            jobs.append((name, t))
            
    if skipped: print(f"\nResume: {skipped} 件の (候補, 被験者) は完了済み -> スキップ")
    if not jobs:
        if args.only_target is None: aggregate_all(args.total, args.fixed, base_dir, manifest, cands)
        return

    if args.jobs is not None: num_workers = max(1, args.jobs)
    elif num_gpus > 0: num_workers = max(1, num_gpus * PROCESSES_PER_GPU)
    else: num_workers = 1   
    num_workers = min(num_workers, len(jobs))
    print(f"\n検出GPU   : {num_gpus} -> {num_workers} 並列 / 全 {len(jobs)} fold\n", flush=True)

    args_list = [(i, name, cands[name], t, kept, args.total, args.fixed, args.epochs,
                  args.teacher_importance, args.ranking, num_gpus, base_dir, cache_dir,
                  args.normalize, args.class_weight,
                  args.gate_feat, args.gate_teacher, args.gate_hints,
                  args.select, args.mask_randomize)
                 for i, (name, t) in enumerate(jobs)]

    if num_workers == 1:
        for a in args_list: worker(*a)
    else:
        with mp.Pool(processes=num_workers) as pool:
            pool.starmap(worker, args_list, chunksize=1)

    if args.only_target is None:
        aggregate_all(args.total, args.fixed, base_dir, manifest, cands)
    else:
        print(f"fold (target={args.only_target}) 完了。")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()