"""VLAの固定電極候補を保存済みfold別Optuna条件のLOSOで比較する。

LOSO法はratio_fixdynamic_eeggate_optunar.pyと同一で、このコード自身はOptuna探索を
行わない。全20名を学習候補とし、target以外の19名でGate + EEGNetを学習して、
manifestで適格な11名だけをouter testにする。提案法の固定電極は全fold共通の
global importance rankingから選ぶ。
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import traceback

# プロジェクトルートを import パスに追加 (select_net / model を解決するため)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use('Agg')  # 並列プロセス / ヘッドレス環境で描画するため非GUIバックエンド
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import warnings
from scipy import stats
from scipy.interpolate import griddata
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report,
                             cohen_kappa_score, f1_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)
from torch.utils.data import DataLoader, TensorDataset

from select_net.select_net_class_optunar import gcn_select_net
from model.EEGNet import CustomEEGNet

warnings.filterwarnings("ignore")

# =========================================================================
# 設定とチャネル定義
# =========================================================================
DATA_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA/processdData/'
            'subject_wise_2class_allsubject')
CHEAT_SHEET_DIR = os.environ.get(
    'CHEAT_SHEET_DIR',
    '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA/'
    'LOSO_pretrainedmodel/LOSO_Optuna_All20_Eligible11_InnerHoldout3')
CHEAT_SHEET_TAG = '_'.join([p for p in CHEAT_SHEET_DIR.rstrip('/').split('/')[-2:] if p])

RESEARCH_BASE_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA/'
                     'research_fix_electrode/'
                     'results_candidates_paper2class_fix4_eeggate_optunar/')

NUM_CLASSES = 2
CLASS_NAMES = ['Awake', 'Fatigue']
CLASS_LABELS = {0: 'Awake', 1: 'Fatigue'}
CLASS_ORDER = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]

# チャネル名 (18ch 10-20)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T3', 'C3', 'Cz', 'C4', 'T4',
                 'T5', 'P3', 'P4', 'T6', 'O1', 'O2']
NUM_CHANNELS = len(CHANNEL_NAMES)
NUM_BANDS = 5
CH_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}

COORDS = {
    'Fp1': (-0.3, 0.9), 'Fp2': (0.3, 0.9),
    'F7': (-0.8, 0.6), 'F3': (-0.4, 0.6), 'Fz': (0, 0.6), 'F4': (0.4, 0.6), 'F8': (0.8, 0.6),
    'T3': (-0.9, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0, 0.0), 'C4': (0.5, 0.0), 'T4': (0.9, 0.0),
    'T5': (-0.8, -0.6), 'P3': (-0.4, -0.6), 'P4': (0.4, -0.6), 'T6': (0.8, -0.6),
    'O1': (-0.3, -0.9), 'O2': (0.3, -0.9),
}

GLOBAL_IMPORTANCE_RANKING = [
    'O2', 'O1', 'Fp1', 'Cz', 'T5', 'P4', 'Fp2', 'C4', 'T3', 'T6', 'T4', 'F4', 'F3', 'F7', 'Fz', 'P3', 'C3', 'F8'
]
assert sorted(GLOBAL_IMPORTANCE_RANKING) == sorted(CHANNEL_NAMES)

DEFAULT_TOTAL_ELECTRODES = 8
DEFAULT_FIXED_ELECTRODES = 4

SEED = 42
BATCH_SIZE = 128
LAMBDA_SPARSITY = 1e-3
FIXED_EPOCHS = 100

NORMALIZE = 'global'
CLASS_WEIGHT = 'none'

OPTUNA_MODEL_PARAM_KEYS = (
    'hidden_size', 'num_hidden_layers', 'transformer_dropout', 'cnn_dropout',
    'gnn_dropout', 'num_attention_heads', 'gnn_heads', 'cnn_out_channels')

GATE_FEAT = 'band'
GATE_TEACHER = 'fold_const'
GATE_HINTS = False
SELECT = 'topk'
MASK_RANDOMIZE = 0.3

PROCESSES_PER_GPU = 4
TEACHER_CACHE_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA/'
                     'research_number_of_electrode/'
                     'results_bestnum_paper2class_eeggate_optunar/teacher_cache')

# 報告する評価指標
METRICS = [('macro_f1', 'Macro F1'),
           ('acc', 'Accuracy(%)'),
           ('balanced_acc_pct', 'Balanced Acc(%)'),
           ('kappa', "Cohen's Kappa")]
METRIC_KEYS = [k for k, _ in METRICS]

FIG_METRICS = [('balanced_acc_pct', 'Balanced Acc(%)'),
               ('macro_f1', 'Macro F1'),
               ('kappa', "Cohen's Kappa")]
FIG_METRICS_ALL = [('acc', 'Accuracy (%)'),
                   ('macro_f1', 'Macro F1'),
                   ('kappa', "Cohen's Kappa")]

REPORT_METRICS = [('acc', 'Accuracy '),
                  ('balanced_acc_pct', 'Balanced Accuracy'),
                  ('macro_precision', 'Precision'),
                  ('macro_recall', 'Recall   '),
                  ('macro_f1', 'F1-Score '),
                  ('kappa', 'Kappa    ')]
PCT_METRICS = {'acc', 'balanced_acc_pct'}

# =========================================================================
# 比較候補 (18ch 10-20)
# =========================================================================
CANDIDATES = {
    'proposed': dict(
        channels=None,
        ranking='global',
        group='proposed',
        rationale='提案手法。全fold共通の重要度ランキングの上位K本を固定する。'),

    'occipital_alpha': dict(
        channels=['O1', 'O2', 'P3', 'P4'], 
        ranking=None,
        group='literature',
        rationale='後頭の左右1対(O1/O2)と頭頂(P3/P4)で、α波の変化を後方広域で捉える構成。'),

    'frontal_midline_theta': dict(
        channels=['Fz', 'F3', 'F4', 'Fp1'], 
        ranking=None,
        group='literature',
        rationale='前頭正中θ。Fzを正中アンカーにし、左右の前頭(F3/F4)と前頭極(Fp1)を加えた構成。'),

    'prefrontal_wearable': dict(
        channels=['Fp1', 'Fp2', 'F7', 'F8'], 
        ranking=None,
        group='literature',
        rationale='前額部〜前頭側部のみで構成した、ウェアラブルデバイスでの実運用を想定した形態。'),

    'central_parietal': dict(
        channels=['Cz', 'C3', 'C4', 'P3'], 
        ranking=None,
        group='literature',
        rationale='感覚運動野を中心に、Czを正中としてC3/C4と頭頂(P3)を加えた構成。'),

    'frontal_occipital': dict(
        channels=['Fz', 'Cz', 'O1', 'O2'], 
        ranking=None,
        group='literature',
        rationale='前頭θと後頭αの増大を同時に取るため、前頭正中(Fz)・中心(Cz)と後頭(O1/O2)で構成。'),

    'temporal_lateral': dict(
        channels=['T3', 'T4', 'T5', 'T6'], 
        ranking=None,
        group='literature',
        rationale='運転疲労の推定に有効とされる両側の側頭部(T3/T4)と後側頭(T5/T6)。'),

    'uniform_coverage': dict(
        channels=['F3', 'F4', 'P3', 'P4'], 
        ranking=None,
        group='literature',
        rationale='前頭と頭頂から均等に左右対称で2本ずつ取った空間サンプリング。'),

    'left_hemisphere': dict(
        channels=['Fp1', 'F3', 'C3', 'O1'], 
        ranking=None,
        group='control',
        rationale='左半球のみ(対照)。前頭極から後頭までの縦鎖を片側だけで取る。'),

    'right_hemisphere': dict(
        channels=['Fp2', 'F4', 'C4', 'O2'], 
        ranking=None,
        group='control',
        rationale='右半球のみ(対照)。left_hemisphere の鏡像。'),
}

GROUP_COLORS = {'proposed': '#c0504d', 'literature': '#3b6fb6', 'control': '#9aa4b1',
                'random': '#7fa66b'}

# =========================================================================
# 0. データ
# =========================================================================
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


def load_optuna_fold_config(target):
    """保存済みのfold別Optuna最良設定を読む（探索は実行しない）。"""
    candidates = [
        os.path.join(CHEAT_SHEET_DIR, f'fold_target_{target}.json'),
        os.path.join(CHEAT_SHEET_DIR, f'inner_target_{target}.json'),
    ]
    config_path = next((p for p in candidates if os.path.exists(p)), None)
    if config_path is None:
        raise FileNotFoundError(
            f'fold {target} のOptuna設定がありません: {CHEAT_SHEET_DIR}')
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    params = config.get('params')
    if not isinstance(params, dict):
        raise ValueError(f'{config_path}: params がありません')
    required = (*OPTUNA_MODEL_PARAM_KEYS, 'learning_rate', 'batch_size',
                'weight_decay')
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f'{config_path}: Optunaパラメータが不足しています: {missing}')
    if config.get('target') not in (None, target):
        raise ValueError(
            f'{config_path}: target={config.get("target")} は要求された {target} と不一致です')
    return config, config_path


def load_optuna_standardizer(target):
    """Optuna教師のinner-trainから保存されたmean/stdを読む。"""
    path = os.path.join(CHEAT_SHEET_DIR, f'standardizer_target_{target}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'fold {target} の標準化統計量がありません: {path}')
    with np.load(path) as values:
        if 'mean' not in values or 'std' not in values:
            raise ValueError(f'{path}: mean/std がありません')
        mean = torch.from_numpy(values['mean'].astype(np.float32, copy=False))
        std = torch.from_numpy(values['std'].astype(np.float32, copy=False))
    return mean, std, path


def load_subject_x(subject_id, normalize):
    x = np.load(os.path.join(DATA_DIR, f'paper_eeg_{subject_id}.npy')).astype(np.float32)
    return normalize_subject(x, normalize)


def load_subjects(subject_ids, normalize):
    xs, ys = [], []
    for s in subject_ids:
        xs.append(load_subject_x(s, normalize))
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
        weights = np.where(
            counts > 0, n / (NUM_CLASSES * np.maximum(counts, 1)), 1.0)
        return torch.FloatTensor(weights).to(device)
    raise ValueError(f'unknown class_weight mode: {mode}')


def make_loader(subject_ids, shuffle, teacher_cache, normalize,
                imp_override=None, generator=None, train_mean=None, train_std=None,
                batch_size=BATCH_SIZE):
    x, y = load_subjects(subject_ids, normalize)
    
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
    
    assert len(hints) == len(x) and len(imp) == len(x)
    ds = TensorDataset(x, y, hints, imp)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)

def find_cheat_sheet(target):
    candidates = [os.path.join(CHEAT_SHEET_DIR, f'model_target{target}.pth'),
                  os.path.join(CHEAT_SHEET_DIR, f'model_target_{target}.pth'),
                  os.path.join(CHEAT_SHEET_DIR, f'best_model_target{target}.pth')]
    return next((p for p in candidates if os.path.exists(p)), None)

def fold_constant_importance(teacher_cache, train_subjects):
    """fold 内で共通に使う教師の電極重要度 [18] を、【学習被験者だけ】から作る。"""
    return np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects],
                          axis=0).mean(axis=0)

def fold_ranking(teacher_cache, train_subjects):
    imp = np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects],
                         axis=0).mean(axis=0)
    order = np.argsort(imp)[::-1]
    return [CHANNEL_NAMES[i] for i in order], imp

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
            continue
        chs = list(chs[:fixed_n])
        out[name] = dict(spec, channels=chs,
                         indices=[CH_INDEX[c] for c in chs])
    return out

def fold_fixed_channels(name, spec, fixed_n, teacher_cache, train_subjects, ranking_mode):
    if spec['channels'] is not None:
        return list(spec['channels'])
    # 提案法は全foldで同じ固定電極を使う。teacher_cacheはGate入力にのみ使用する。
    return list(GLOBAL_IMPORTANCE_RANKING[:fixed_n])

def candidate_dir(base_dir, total, fixed, name):
    return os.path.join(base_dir, f'total{total}_fix{fixed}', f'cand_{name}')

# =========================================================================
# 1. Models & Gate Mechanism
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

def cls_row_importance(attention_weights):
    rows = []
    for layer in attention_weights:
        a = torch.stack(layer)
        rows.append(a.mean(dim=0)[:, 0, 1:])
    return torch.stack(rows).mean(dim=0)

@torch.no_grad()
def teacher_outputs(cheat_sheet_model, x, mode='col'):
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

def teacher_cache_path(cache_dir, mode, target, normalize):
    return os.path.join(
        cache_dir, f'teacher_{mode}_{prep_tag(normalize)}_target{target}.npz')

@torch.no_grad()
def build_teacher_cache(target, subjects, mode, device, cache_dir, normalize):
    """fold固有のOptuna済み教師の出力を計算して保存する。"""
    path = teacher_cache_path(cache_dir, mode, target, normalize)
    if os.path.exists(path):
        return path

    cheat_path = find_cheat_sheet(target)
    if cheat_path is None:
        raise FileNotFoundError(
            f'fold {target} のOptuna教師モデルがありません: {CHEAT_SHEET_DIR}')

    optuna_config, config_path = load_optuna_fold_config(target)
    optuna_params = optuna_config['params']
    model_kwargs = {key: optuna_params[key] for key in OPTUNA_MODEL_PARAM_KEYS}
    model = gcn_select_net(num_classes=NUM_CLASSES, **model_kwargs).to(device)
    model.load_state_dict(torch.load(cheat_path, map_location=device))
    model.eval()

    # 教師には教師学習時のinner-train統計を適用する。
    train_mean, train_std, standardizer_path = load_optuna_standardizer(target)
    teacher_batch_size = int(optuna_params.get('batch_size', BATCH_SIZE))
    out = {}
    for s in subjects:
        x, _ = load_subjects([s], 'none')
        x = (x - train_mean) / (train_std + 1e-8)
        hints, imps = [], []
        for i in range(0, len(x), teacher_batch_size):
            h, m = teacher_outputs(
                model, x[i:i + teacher_batch_size].to(device), mode)
            hints.append(h.cpu())
            imps.append(m.cpu())
        out[f'hints_{s}'] = torch.cat(hints).numpy()
        out[f'imp_{s}'] = torch.cat(imps).numpy()
    out['optuna_params_json'] = np.asarray(
        json.dumps(optuna_params, sort_keys=True), dtype=str)
    out['optuna_config_path'] = np.asarray(config_path, dtype=str)
    out['standardizer_path'] = np.asarray(standardizer_path, dtype=str)

    tmp = path + '.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, path)

    del model
    torch.cuda.empty_cache()
    return path

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
# 2. 1 fold の学習・評価
# =========================================================================
@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, n_batch = 0.0, 0
    trues, preds, probs = [], [], []

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
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

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
        'balanced_acc_pct': 100.0 * balanced_accuracy_score(trues, preds),
        'macro_precision': precision_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_recall': recall_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_f1': f1_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'avg_selected_electrodes': usage_sum / max(n_seen, 1),
        'n_class0_true': int(sum(1 for t in trues if t == 0)),
        'n_class1_true': int(sum(1 for t in trues if t == 1)),
        'both_classes_present': len(set(trues)) == NUM_CLASSES,
    }
    return metrics, trues, preds, probs, gate_sum, gate_counts

def run_fold(name, spec, target, kept, total, fixed, epochs,
             ranking_mode, teacher_importance, device, cand_dir, cache_dir,
             normalize=NORMALIZE, class_weight=CLASS_WEIGHT,
             gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER,
             gate_hints=GATE_HINTS, select=SELECT,
             mask_randomize=MASK_RANDOMIZE):
    train_subjects = [s for s in kept if s != target]
    optuna_config, optuna_config_path = load_optuna_fold_config(target)
    _, _, teacher_standardizer_path = load_optuna_standardizer(target)
    learning_rate = float(optuna_config['params']['learning_rate'])
    batch_size = int(optuna_config['params']['batch_size'])
    weight_decay = float(optuna_config['params']['weight_decay'])
    if learning_rate <= 0:
        raise ValueError(
            f'{optuna_config_path}: learning_rate は正の値が必要です: {learning_rate}')
    if batch_size <= 0:
        raise ValueError(
            f'{optuna_config_path}: batch_size は正の整数が必要です: {batch_size}')
    if weight_decay < 0:
        raise ValueError(
            f'{optuna_config_path}: weight_decay は0以上が必要です: {weight_decay}')

    seed = SEED + target
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator()
    gen.manual_seed(seed)

    tag = f'[{name}/Sub{target}]'

    # 学生はtargetを除くouter-train全体から標準化統計を計算する。
    train_mean, train_std = None, None
    if normalize == 'global':
        x_train_raw, _ = load_subjects(train_subjects, 'none')
        train_mean = x_train_raw.mean(dim=0, keepdim=True)
        train_std = x_train_raw.std(dim=0, keepdim=True)
        del x_train_raw

    cache_path = build_teacher_cache(
        target, kept, teacher_importance, device, cache_dir, normalize)
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

    print(f"{tag} train {len(train_subjects)}名 / test [{target}] "
          f"| fixed={fixed_names} | var_target={target_var_electrodes}"
          f"/{len(variable_indices)} | epochs={epochs} "
          f"| normalize={normalize} | class_weight={class_weight} "
          f"| lr_gate=lr_eegnet={learning_rate:.6g} | batch={batch_size} "
          f"| wd={weight_decay:.6g} | select={select} | {device}", flush=True)
    print(f"{tag} fixed ranking: global -> {fixed_names} | "
          f"cheat sheet: {os.path.basename(find_cheat_sheet(target))}", flush=True)

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
        {'params': model.gate.parameters(), 'lr': learning_rate},
        {'params': model.classifier.parameters(), 'lr': learning_rate},
    ], weight_decay=weight_decay)
    
    weights = class_weights(train_subjects, class_weight, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_loader = make_loader(train_subjects, shuffle=True, teacher_cache=teacher_cache,
                               normalize=normalize, imp_override=imp_override,
                               generator=gen, train_mean=train_mean,
                               train_std=train_std, batch_size=batch_size)

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
                              normalize=normalize, imp_override=imp_override,
                              train_mean=train_mean, train_std=train_std,
                              batch_size=batch_size)

    tm, trues, preds, probs, gate_sum, gate_counts = evaluate(model, test_loader, device, loss_fn)

    labels = list(range(NUM_CLASSES))
    per_rec = recall_score(trues, preds, labels=labels, average=None, zero_division=0)
    per_f1 = f1_score(trues, preds, labels=labels, average=None, zero_division=0)
    tm['kappa'] = float(cohen_kappa_score(trues, preds, labels=labels))
    if np.isnan(tm['kappa']):
        tm['kappa'] = 0.0
    for i, c in enumerate(CLASS_ORDER):
        tm[f'recall_{c}'] = float(per_rec[i])
        tm[f'f1_{c}'] = float(per_f1[i])

    print(f"{tag} test 覚醒{tm['n_class0_true']}/疲労{tm['n_class1_true']} -> "
          f"Acc {tm['acc']:.2f}% | Prec {tm['macro_precision']:.4f} "
          f"| BalAcc {tm['balanced_acc_pct']:.2f}% "
          f"| Rec {tm['macro_recall']:.4f} | F1 {tm['macro_f1']:.4f} "
          f"| 選定 {tm['avg_selected_electrodes']:.2f}ch (目標 {total}ch)", flush=True)

    result = {**tm, 'candidate': name, 'group': spec['group'], 'target': target,
              'n_total_electrodes': total, 'n_fixed_electrodes': fixed,
              'n_variable_target': target_var_electrodes,
              'n_variable_channels': len(variable_indices),
              'fixed_channels': fixed_names, 'ranking_mode': 'global',
              'n_test': len(trues), 'train_subjects': train_subjects, 'epochs': epochs,
              'teacher_importance': teacher_importance,
              'teacher_dir': CHEAT_SHEET_DIR, 'normalize': normalize,
              'class_weight': class_weight,
              'gate_feat': gate_feat, 'gate_teacher': gate_teacher,
              'gate_hints': gate_hints, 'select': select,
              'mask_randomize': mask_randomize,
              'lr_gate': learning_rate, 'lr_classifier': learning_rate,
              'batch_size': batch_size, 'weight_decay': weight_decay,
              'teacher_optuna_params': optuna_config['params'],
              'teacher_optuna_score': optuna_config.get(
                  'score', optuna_config.get('inner_score')),
              'teacher_optuna_trial': optuna_config.get(
                  'trial', optuna_config.get('optuna_trial')),
              'teacher_optuna_epoch': optuna_config.get(
                  'epochs', optuna_config.get('epoch')),
              'teacher_optuna_config': optuna_config_path,
              'teacher_standardizer': teacher_standardizer_path}

    with open(os.path.join(cand_dir, f'fold_target{target}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(cand_dir, f'fold_target{target}.npz'),
             trues=np.array(trues), preds=np.array(preds),
             probs=np.array(probs, dtype=np.float32),
             gate_sum=gate_sum.numpy(), gate_counts=gate_counts.numpy(),
             fixed_indices=np.array(fixed_indices, dtype=np.int64),
             curve_train_loss=np.array(curve_train_loss),
             curve_train_acc=np.array(curve_train_acc),
             curve_usage=np.array(curve_usage))
    torch.save(model.state_dict(),
               os.path.join(cand_dir, f'concept_dynamic_net_target{target}.pth'))

    del model, classifier_model, optimizer
    torch.cuda.empty_cache()
    return result

def worker(job_idx, name, spec, target, kept, total, fixed, epochs,
           ranking_mode, teacher_importance, num_gpus, base_dir, cache_dir,
           normalize=NORMALIZE, class_weight=CLASS_WEIGHT,
           gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER,
           gate_hints=GATE_HINTS, select=SELECT,
           mask_randomize=MASK_RANDOMIZE):
    gpu_id = job_idx % num_gpus if num_gpus > 0 else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    cand_dir = candidate_dir(base_dir, total, fixed, name)
    os.makedirs(cand_dir, exist_ok=True)

    try:
        return run_fold(name, spec, target, kept, total, fixed, epochs,
                        ranking_mode, teacher_importance, device, cand_dir, cache_dir,
                        normalize, class_weight, gate_feat, gate_teacher,
                        gate_hints, select, mask_randomize)
    except Exception as e:
        print(f"[{name}/Sub{target}] FAILED: {e}", flush=True)
        traceback.print_exc()
        return None

# =========================================================================
# 3. 候補ごとの集約・可視化
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
    fig.suptitle(f'Electrode Selection (Total {total}ch / Fixed {num_fixed}ch, mean of {n_folds} folds)', fontsize=15)
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
    plt.title(f'Electrode Selection Rate (Total {total}ch / Fixed {num_fixed}ch, {n_folds} folds)')
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
    files = sorted(glob.glob(os.path.join(cand_dir, 'fold_target*.json')))
    if not files: return None

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
        'balanced_accuracy': balanced_accuracy_score(global_true, global_pred),
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

    plot_confusion(global_true, global_pred,
                   os.path.join(cand_dir, 'confusion_matrix_pooled.png'),
                   f'{name} (total {total}ch / fixed {fixed}ch) pooled LOSO '
                   f'(Acc: {pooled["accuracy"]*100:.2f}%, F1: {pooled["macro_f1"]:.4f})')
    plot_class_topography(class_distribution,
                          os.path.join(cand_dir, 'class_wise_topography.png'),
                          total, fixed, len(targets))
    plot_selection_bar(overall_usage, fixed_rate,
                       os.path.join(cand_dir, 'electrode_selection_rate.png'),
                       total, fixed, len(targets))

    learning_rates = [float(fold_results[s]['lr_classifier']) for s in targets]
    batch_sizes = [int(fold_results[s]['batch_size']) for s in targets]
    weight_decays = [float(fold_results[s]['weight_decay']) for s in targets]
    with open(os.path.join(cand_dir, 'loso_report.txt'), 'w') as f:
        f.write(f"===== LOSO 2-class (VLA Optuna teacher) / candidate '{name}' "
                f"[{spec['group']}] / total {total}ch, fixed {fixed}ch =====\n")
        f.write(f"rationale: {spec['rationale']}\n")
        f.write("metrics: Accuracy / Balanced Accuracy / Precision / Recall / F1-Score / Kappa "
                "(Precision, Recall, F1 は macro 平均)\n")
        f.write(f"epochs : {fold_results[targets[0]].get('epochs')} (全fold固定)\n")
        f.write(f"labeling: {manifest['labeling_rule']}\n")
        f.write("\n===== Hyperparameters =====\n")
        f.write("  fixed_ranking:     global (all folds use the same ranking)\n")
        f.write("  lr_gate/lr_classifier: fold-specific Optuna learning_rate\n")
        f.write(f"  learning_rate range: {min(learning_rates):.8g} - "
                f"{max(learning_rates):.8g}\n")
        f.write(f"  batch_size values: {sorted(set(batch_sizes))}\n")
        f.write(f"  weight_decay range: {min(weight_decays):.8g} - "
                f"{max(weight_decays):.8g}\n")
        f.write(f"  teacher_dir:       {CHEAT_SHEET_DIR}\n")
        f.write(f"  normalization:     {fold_results[targets[0]].get('normalize')}\n")
        f.write(f"  total_electrodes:  {total}\n")
        f.write(f"  fixed_electrodes:  {fixed}\n")
        f.write(f"  variable_target:   {total - fixed} "
                f"(可変チャネル {fold_results[targets[0]].get('n_variable_channels')} 本から選ぶ)\n\n")
        for title, summ in [('全ターゲット', summary_all),
                            ('両クラスを持つ被験者のみ', summary_balanced)]:
            if summ is None: continue
            f.write(f"----- {title} (n={summ['n_subjects']}) -----\n")
            for key, label in REPORT_METRICS:
                v = summ[key]
                if v is None: continue
                pct = key in PCT_METRICS
                unit, fmt = ('%', '.2f') if pct else ('', '.4f')
                f.write(f"  {label}: {v['mean']:{fmt}}{unit} ± {v['std']:{fmt}}{unit}\n")
            for c in CLASS_ORDER:
                v = summ[f'recall_{c}']
                f.write(f"  recall[{c}]: {v['mean']:.4f} ± {v['std']:.4f}\n")
            sel = summ['avg_selected_electrodes']
            f.write(f"  Electrodes: {sel['mean']:.2f} ± {sel['std']:.2f} ch (目標 {total}ch)\n\n")

    out = {
        'candidate': name,
        'group': spec['group'],
        'rationale': spec['rationale'],
        'n_total_electrodes': total,
        'n_fixed_electrodes': fixed,
        'n_folds': len(targets),
        'targets': targets,
        'degenerate_targets': degenerate,
        'summary_all_targets': summary_all,
        'summary_balanced_targets': summary_balanced,
        'pooled': pooled,
        'electrode_selection_rate': {CHANNEL_NAMES[i]: float(overall_usage[i]) for i in range(NUM_CHANNELS)},
        'fixed_rate_across_folds': {CHANNEL_NAMES[i]: float(fixed_rate[i]) for i in range(NUM_CHANNELS)},
        'fixed_channels_per_fold': {str(s): fold_results[s]['fixed_channels'] for s in targets},
        'per_subject': {str(s): fold_results[s] for s in targets},
    }
    with open(os.path.join(cand_dir, 'loso_results.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n===== candidate '{name}' [{spec['group']}] ({len(targets)} folds) =====")
        for title, summ in [('全ターゲット', summary_all), ('両クラスを持つ被験者のみ', summary_balanced)]:
            if summ is None: continue
            print(f" {title} (n={summ['n_subjects']}名)")
            for key, label in REPORT_METRICS:
                v = summ[key]
                if v is None: continue
                pct = key in PCT_METRICS
                unit, fmt = ('%', '.2f') if pct else ('', '.4f')
                print(f"   {label} : {v['mean']:{fmt}}{unit} ± {v['std']:{fmt}}{unit}")
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
        if np.allclose(d, 0):
            p_t = p_w = 1.0
        else:
            p_t = float(stats.ttest_rel(a, b).pvalue)
            try:
                p_w = float(stats.wilcoxon(a, b).pvalue)
            except Exception:
                p_w = float('nan')
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
        per_fold[name] = {int(s): v for s, v in r['per_subject'].items()}

    if not details: return

    rows = []
    for r in details:
        sa, sb = r['summary_all_targets'], r['summary_balanced_targets']
        def g(summ, key, stat='mean'):
            if summ is None or summ.get(key) is None: return float('nan')
            return round(float(summ[key][stat]), 6)
        row = {
            'candidate': r['candidate'], 'group': r['group'],
            'n_total_electrodes': total, 'n_fixed_electrodes': fixed,
            'n_folds': r['n_folds'],
            'fixed_channels': ';'.join(sorted(set(c for v in r['fixed_channels_per_fold'].values() for c in v))
                                       if r['candidate'].startswith('proposed') else next(iter(r['fixed_channels_per_fold'].values()))),
            'avg_selected_electrodes': g(sa, 'avg_selected_electrodes'),
            'all_acc_mean': g(sa, 'acc'), 'all_acc_std': g(sa, 'acc', 'std'),
            'all_bacc_mean': g(sa, 'balanced_acc_pct'),
            'all_bacc_std': g(sa, 'balanced_acc_pct', 'std'),
            'all_precision_mean': g(sa, 'macro_precision'),
            'all_recall_mean': g(sa, 'macro_recall'),
            'all_f1_mean': g(sa, 'macro_f1'), 'all_f1_std': g(sa, 'macro_f1', 'std'),
            'all_kappa_mean': g(sa, 'kappa'),
            'bal_acc_mean': g(sb, 'acc'), 'bal_acc_std': g(sb, 'acc', 'std'),
            'bal_bacc_mean': g(sb, 'balanced_acc_pct'),
            'bal_bacc_std': g(sb, 'balanced_acc_pct', 'std'),
            'bal_precision_mean': g(sb, 'macro_precision'),
            'bal_recall_mean': g(sb, 'macro_recall'),
            'bal_f1_mean': g(sb, 'macro_f1'), 'bal_f1_std': g(sb, 'macro_f1', 'std'),
            'bal_kappa_mean': g(sb, 'kappa'),
            'pooled_acc': round(r['pooled']['accuracy'] * 100.0, 4),
            'pooled_bacc': round(r['pooled']['balanced_accuracy'] * 100.0, 4),
            'pooled_f1': round(r['pooled']['macro_f1'], 6),
            'pooled_kappa': round(r['pooled']['kappa'], 6),
        }
        for c in CLASS_ORDER: row[f'bal_recall_{c}'] = g(sb, f'recall_{c}')
        rows.append(row)

    rows.sort(key=lambda x: (-x['bal_f1_mean'] if not np.isnan(x['bal_f1_mean']) else 0))
    csv_path = os.path.join(res_dir, 'summary_candidates.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    add = lines.append
    add("===== Candidate benchmark: proposed vs. fatigue-motivated fixed electrode sets =====")
    add(f"data      : {DATA_DIR}")
    add(f"protocol  : LOSO / total {total}ch のうち {fixed}ch を固定、残り {total-fixed}ch を Gate が動的選定")
    add(f"norm      : {NORMALIZE} (outer-train statistics for the student)")
    add("fixed rank: global (same proposed fixed channels in every fold)")
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

    add("--- Mean ± SD over LOSO folds (all targets) ---")
    add(hdr); add("-" * len(hdr))
    for name in order:
        r = stat_of[name]
        sa = r['summary_all_targets']
        s = f"{name:<22} {r['group']:<11}"
        for key, _ in METRICS:
            v = sa.get(key) if sa else None
            s += "              n/a" if v is None else f"{v['mean']:>10.4f}±{v['std']:.3f}"
        add(s)
    add("")

    for subset_name, subset in [('balanced targets', balanced), ('all targets', details[0]['targets'])]:
        add(f"--- Paired comparison against 'proposed' ({subset_name}, same subjects) ---")
        for key, label in METRICS:
            add(f"  [{label}]")
            for name, dm, ds, n, p_t, p_w in paired_table(per_fold, order, subset, key):
                if np.isnan(dm):
                    add(f"    {name:<22} (被験者数不足で検定不可, n={n})")
                    continue
                sig = '**' if min(p_t, p_w) < 0.01 else ('*' if min(p_t, p_w) < 0.05 else '')
                add(f"    {name:<22} delta={dm:+.4f} ±{ds:.4f} (n={n}, p_t={p_t:.3f}, p_w={p_w:.3f}){sig}")
            add("")

    txt = "\n".join(lines)
    with open(os.path.join(res_dir, 'candidate_summary.txt'), 'w') as f: f.write(txt + "\n")
    print("\n" + txt)

    # 図1: 主指標のプロット
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    def plot_metric_panels(metrics, summ_key, subjects_of, out_name, scope_label):
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 5.0))
        for ax, (key, label) in zip(np.atleast_1d(axes), metrics):
            vals = [(n, (stat_of[n][summ_key] or {}).get(key)) for n in order]
            vals = [(n, v) for n, v in vals if v is not None]
            vals.sort(key=lambda kv: kv[1]['mean'])
            names = [n for n, _ in vals]
            means = np.array([v['mean'] for _, v in vals])
            sds = np.array([v['std'] for _, v in vals])
            cols = [GROUP_COLORS.get(stat_of[n]['group'], '#888') for n in names]
            y = np.arange(len(names))
            ax.barh(y, means, xerr=sds, color=cols, alpha=.9, error_kw=dict(lw=1, capsize=2.5, ecolor='#444'))
            for yi, n in zip(y, names):
                pts = [per_fold[n][t][key] for t in subjects_of(n) if t in per_fold[n]]
                ax.scatter(pts, np.full(len(pts), yi, dtype=float), s=8, color='black', alpha=.45, zorder=3)
            ax.set_yticks(y, names)
            lo = max(0.0, float(means.min() - 4 * (sds.max() + 1e-3)))
            ax.set_xlim(lo, float(means.max() + 2 * (sds.max() + 1e-3)))
            ax.set_xlabel(label)
            ax.set_title(label)
        fig.suptitle(f'Proposed vs. fatigue-motivated fixed electrode sets '
                     f'(total {total}ch / fixed {fixed}ch, LOSO {scope_label})', y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(res_dir, out_name))
        plt.close(fig)

    plot_metric_panels(FIG_METRICS, 'summary_balanced_targets', lambda n: balanced, 'fig_candidates_metrics.png', f'balanced targets n={len(balanced)}')
    plot_metric_panels(FIG_METRICS_ALL, 'summary_all_targets', lambda n: stat_of[n]['targets'], 'fig_candidates_metrics_alltargets.png', f"all targets n={len(details[0]['targets'])}")

    with open(os.path.join(res_dir, 'summary_candidates.json'), 'w') as f:
        json.dump({'data_dir': DATA_DIR,
                   'cheat_sheet_dir': CHEAT_SHEET_DIR,
                   'n_total_electrodes': total,
                   'n_fixed_electrodes': fixed,
                   'labeling_rule': manifest['labeling_rule'],
                   'excluded_subjects': manifest['excluded_subjects'],
                   'kept_subjects': manifest['kept_subjects'],
                   'all_subjects': manifest.get('all_subjects', manifest['kept_subjects']),
                   'donor_only_subjects': manifest.get('donor_only_subjects', []),
                   'balanced_targets': balanced,
                   'candidates': [d['candidate'] for d in details],
                   'per_candidate': details}, f, indent=2, ensure_ascii=False)

    valid = [r for r in rows if not np.isnan(r['bal_f1_mean'])]
    if valid:
        best = max(valid, key=lambda x: x['bal_f1_mean'])
        print(f"\n>>> Best by balanced-target macro F1: '{best['candidate']}' "
              f"[{best['group']}] (F1={best['bal_f1_mean']:.4f} ± {best['bal_f1_std']:.4f})")

# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total', type=int, default=DEFAULT_TOTAL_ELECTRODES,
                    help=f'総電極数。既定 {DEFAULT_TOTAL_ELECTRODES}')
    ap.add_argument('--fixed', type=int, default=DEFAULT_FIXED_ELECTRODES,
                    help=f'固定電極数。既定 {DEFAULT_FIXED_ELECTRODES}')
    ap.add_argument('--epochs', type=int, default=FIXED_EPOCHS,
                    help=f'学習エポック数（既定 {FIXED_EPOCHS}）')
    ap.add_argument('--only', nargs='+', default=None,
                    help='指定した候補名だけ実行する')
    ap.add_argument('--only-candidate', type=str, default=None,
                    help='この候補だけ実行する')
    ap.add_argument('--only-target', type=int, default=None,
                    help='この被験者1名のfoldだけ実行する')
    ap.add_argument('--targets', choices=['eligible', 'balanced'], default='eligible',
                    help='outer testは適格11名のみ。balancedは旧名のalias')
    ap.add_argument('--limit-folds', type=int, default=None)
    ap.add_argument('--random-controls', type=int, default=0)
    ap.add_argument('--teacher-importance', choices=['col', 'cls'], default='col',
                    help='Gateに渡す教師の電極重要度 (既定 col)')
    ap.add_argument('--normalize', choices=['none', 'subject', 'channel_band', 'global'],
                    default=NORMALIZE,
                    help=f'学生モデル入力の標準化方式（既定 {NORMALIZE}）')
    ap.add_argument('--class-weight', choices=['none', 'balanced'],
                    default=CLASS_WEIGHT,
                    help=f'CrossEntropyLossのクラス重み（既定 {CLASS_WEIGHT}）')
    ap.add_argument('--gate-feat', choices=['band', 'mean'], default=GATE_FEAT)
    ap.add_argument('--gate-teacher', choices=['fold_const', 'per_sample', 'none'],
                    default=GATE_TEACHER)
    ap.add_argument('--gate-hints', action='store_true', default=GATE_HINTS)
    ap.add_argument('--select', choices=['topk', 'threshold'], default=SELECT)
    ap.add_argument('--mask-randomize', type=float, default=MASK_RANDOMIZE)
    ap.add_argument('--ranking', choices=['global'], default='global',
                    help="候補 'proposed' は全fold共通global rankingで固定（既定）")
    ap.add_argument('--cache-dir', type=str, default=TEACHER_CACHE_DIR)
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--aggregate-only', action='store_true')
    ap.add_argument('--save-path', type=str, default=RESEARCH_BASE_DIR)
    args = ap.parse_args()

    base_dir = args.save_path
    os.makedirs(base_dir, exist_ok=True)

    if not (1 <= args.total <= NUM_CHANNELS):
        ap.error(f'--total は1〜{NUM_CHANNELS}を指定してください')
    if not (0 <= args.fixed <= args.total):
        ap.error('--fixed は0〜--totalを指定してください')

    manifest = json.load(open(os.path.join(DATA_DIR, 'paper_manifest.json')))
    all_subjects = manifest.get('all_subjects', manifest['kept_subjects'])
    eligible = manifest['loso_target_subjects']

    cands = resolve_candidates(args.fixed, args.random_controls, args.ranking)
    only = list(args.only or [])
    if args.only_candidate: only.append(args.only_candidate)
    if only: cands = {k: v for k, v in cands.items() if k in only}

    if args.aggregate_only:
        aggregate_all(args.total, args.fixed, base_dir, manifest, cands)
        return

    if args.only_target is not None:
        if args.only_target not in eligible:
            raise SystemExit(
                f'--only-target {args.only_target} はtest/validation適格被験者では'
                f'ありません。候補: {eligible}')
        targets = [args.only_target]
    else:
        targets = list(eligible)
        if args.limit_folds: targets = targets[:args.limit_folds]

    num_gpus = torch.cuda.device_count()

    print(f"データ    : {DATA_DIR}")
    print(f"Optuna教師: {CHEAT_SHEET_DIR} (foldごとの保存済み結果を使用)")
    print(f"全学習候補: {len(all_subjects)}名 {all_subjects}")
    print(f"test適格  : {len(eligible)}名 {eligible}")
    print(f"学習提供元のみ: {manifest.get('donor_only_subjects', [])}")
    print(f"今回のtarget: {len(targets)}名 {targets}")
    print(f"総電極数  : {args.total} / 固定 {args.fixed} / 動的 {args.total - args.fixed}")
    print("Optuna    : 探索は実行せず、保存済みfold別結果のみ利用")
    print("最適化設定: foldごとのOptuna learning_rate / batch_size / weight_decayを使用")
    print(f"標準化    : {args.normalize}")
    print(f"固定順    : global -> {GLOBAL_IMPORTANCE_RANKING[:args.fixed]}")

    missing = [t for t in targets if find_cheat_sheet(t) is None]
    if missing:
        raise SystemExit(
            f'\n【エラー】Optuna教師モデルが見つからない被験者: {missing}\n'
            f'  {CHEAT_SHEET_DIR} に model_target_<ID>.pth が必要です。')
    try:
        for t in targets:
            load_optuna_fold_config(t)
            load_optuna_standardizer(t)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f'\n【エラー】Optuna教師の出力一式を確認してください:\n  {exc}') from exc

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for t in targets:
        p = teacher_cache_path(
            cache_dir, args.teacher_importance, t, args.normalize)
        existed = os.path.exists(p)
        build_teacher_cache(t, all_subjects, args.teacher_importance,
                            cache_device, cache_dir, args.normalize)
        print(f"  cache target{t}: {'(既存)' if existed else '(新規作成)'}", flush=True)

    jobs = []
    for name in cands:
        for t in targets:
            done = os.path.join(candidate_dir(base_dir, args.total, args.fixed, name), f'fold_target{t}.json')
            if not args.redo and os.path.exists(done): continue
            jobs.append((name, t))
            
    if not jobs:
        print("実行すべき fold がありません。集約に進みます。")
        if args.only_target is None: aggregate_all(args.total, args.fixed, base_dir, manifest, cands)
        return

    num_workers = args.jobs if args.jobs is not None else (max(1, num_gpus * PROCESSES_PER_GPU) if num_gpus > 0 else 1)
    num_workers = min(num_workers, len(jobs))
    print(f"\n検出GPU   : {num_gpus} -> {num_workers} 並列 / 全 {len(jobs)} fold\n", flush=True)

    args_list = [(i, name, cands[name], t, all_subjects, args.total, args.fixed,
                  args.epochs, args.ranking, args.teacher_importance, num_gpus,
                  base_dir, cache_dir, args.normalize, args.class_weight,
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

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
