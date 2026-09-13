"""SEED-VIG 2クラス分類の outer LOSO + inner HoldOut + Optuna 実験。

outer LOSO の target セッションは最終評価にだけ使用する。各 outer fold の学習セッションを
さらにセッション単位の inner train/validation に1回だけ分け、Optuna でハイパー
パラメータを選択する。既定では全23セッションを outer target にし、各foldを
inner train 19セッション / validation 3セッション / outer test 1セッションとする。入力標準化の
mean/std は inner train だけから計算し、validation/test に同じ統計量を適用する。

処理順:
  outer target隔離 -> inner train/validation固定分割 -> Optuna
  最良条件でinner trainを再学習 -> validation最良checkpoint -> targetを1回だけ評価

--multi-gpu指定時は1回の起動で親schedulerを立ち上げ、nvidia-smiで空きGPU
メモリを監視しながら、独立したouter fold workerをGPUごとの上限まで割り当てる。
"""

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
import time
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import warnings
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score,
                             cohen_kappa_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# プロジェクトルートへのパスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# モデルとして gcn_select_net をimport
from select_channel.select_net_class_optunar import gcn_select_net

warnings.filterwarnings("ignore")

# =========================================================================
# 設定
# =========================================================================
DATA_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/'
            'processedData/subject_wise_2class')
SAVE_PATH = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/optunar_LOSO/'
             'LOSO_GCN_Optuna_InnerHoldout3_SEEDVIG/')

NUM_CLASSES = 2
CLASS_NAMES = ['Awake', 'Fatigue']

# SEED-VIG の chn 配列から CPZ を除いた17ch
CHANNEL_NAMES = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
                 'CP1', 'CP2', 'P1', 'PZ', 'P2',
                 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']
NUM_CHANNELS = len(CHANNEL_NAMES)

# LOSO対象の全23セッション。被験者番号だけではなく、計測セッションを
# 1つのdomainとして扱う（被験者4と5はそれぞれ2セッションを含む）。
SUBJECT_LIST = [
    '1_20151124_noon_2', '2_20151106_noon', '3_20151024_noon',
    '4_20151105_noon', '4_20151107_noon', '5_20141108_noon',
    '5_20151012_night', '6_20151121_noon', '7_20151015_night',
    '8_20151022_noon', '9_20151017_night', '10_20151125_noon',
    '11_20151024_night', '12_20150928_noon', '13_20150929_noon',
    '14_20151014_night', '15_20151126_night', '16_20151128_night',
    '17_20150925_noon', '18_20150926_noon', '19_20151114_noon',
    '20_20151129_night', '21_20151016_noon',
]
SUBJECT_INDEX = {subject: index for index, subject in enumerate(SUBJECT_LIST)}

# 単一クラスのセッションは全foldの学習には残す一方、balanced指定時の
# outer targetから除く。only_LOSO_vig_2.py と同じ判定基準。
MIN_MINORITY_RATIO = 0.02

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.03
SEED = 42

FIXED_EPOCHS = 100
N_TRIALS = 40
INNER_VAL_SUBJECTS = 3
EARLY_STOP_PATIENCE = 20
PRUNER_WARMUP_EPOCHS = 10

# multi-GPU scheduler defaults. workerごとのメモリ予約値とGPUに残す安全余裕を
# 満たす範囲で、1 GPUにつき最大3 outer fold workerを起動する。
MIN_FREE_GPU_MEMORY_MB = 6000
GPU_MEMORY_RESERVE_MB = 2000
MAX_WORKERS_PER_GPU = 3
GPU_POLL_SECONDS = 30.0
GPU_STATUS_SECONDS = 300.0

# TUAB版と同じモデル・optimizer探索・DataLoader側の探索対象。
DEFAULT_PARAMS = {
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'batch_size': BATCH_SIZE,
    'hidden_size': 64,
    'num_hidden_layers': 4,
    'transformer_dropout': 0.3,
    'cnn_dropout': 0.3,
    'gnn_dropout': 0.3,
    'num_attention_heads': 16,
    'gnn_heads': 4,
    'cnn_out_channels': 16,
    'use_lr_scheduler': False,
}
MODEL_PARAM_KEYS = (
    'hidden_size', 'num_hidden_layers', 'transformer_dropout', 'cnn_dropout',
    'gnn_dropout', 'num_attention_heads', 'gnn_heads', 'cnn_out_channels'
)

# レポート出力用の指標
REPORT_METRICS = [('acc', 'Accuracy '),
                  ('balanced_acc_pct', 'Balanced Acc'),
                  ('macro_precision', 'Precision'),
                  ('macro_recall', 'Recall   '),
                  ('macro_f1', 'F1-Score '),
                  ('kappa', 'Kappa    ')]
PCT_METRICS = {'acc', 'balanced_acc_pct'}

# 頭部トポマップ上のおおよその位置 (SEED-VIG 17ch)
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
# データ処理
# =========================================================================
def build_manifest():
    """配置済みSEED-VIGデータを検査し、実行対象を決定する。"""
    kept, balanced, missing, counts = [], [], [], {}
    for subject in SUBJECT_LIST:
        eeg_path = os.path.join(DATA_DIR, f'eeg_{subject}.npy')
        label_path = os.path.join(DATA_DIR, f'label_{subject}.npy')
        if not (os.path.exists(eeg_path) and os.path.exists(label_path)):
            missing.append(subject)
            continue

        y = np.load(label_path, mmap_mode='r')
        n_awake = int((y == 0).sum())
        n_fatigue = int((y == 1).sum())
        counts[subject] = {
            'n': int(len(y)),
            'n_awake': n_awake,
            'n_fatigue': n_fatigue,
        }
        kept.append(subject)
        if min(n_awake, n_fatigue) >= MIN_MINORITY_RATIO * len(y):
            balanced.append(subject)

    return {
        'kept_subjects': kept,
        'loso_target_subjects': balanced,
        'labeling_rule': 'PERCLOS < 0.35 -> 0 (Awake), >= 0.35 -> 1 (Fatigue)',
        'excluded_subjects': missing,
        'class_counts': counts,
    }


def load_subjects(subject_ids, return_sid=False):
    """被験者ごとの個別標準化やマスク処理を行わず、生データをそのまま結合する"""
    xs, ys, sids = [], [], []
    for s in subject_ids:
        x = np.load(os.path.join(DATA_DIR, f'eeg_{s}.npy')).astype(np.float32)
        y = np.load(os.path.join(DATA_DIR, f'label_{s}.npy'))

        if x.ndim != 4 or x.shape[-1] != NUM_CHANNELS:
            raise ValueError(
                f'{s}: EEG shapeは[N, time, freq, {NUM_CHANNELS}]を期待しますが、'
                f'{x.shape}でした')
        if len(x) != len(y):
            raise ValueError(f'{s}: EEG={len(x)}件、label={len(y)}件で長さが不一致です')
        invalid_labels = np.setdiff1d(np.unique(y), np.arange(NUM_CLASSES))
        if invalid_labels.size:
            raise ValueError(f'{s}: 不正なラベルがあります: {invalid_labels.tolist()}')
            
        xs.append(x)
        ys.append(y)
        sids.append(np.full(len(y), SUBJECT_INDEX[s], dtype=np.int64))
        
    x = torch.FloatTensor(np.concatenate(xs, axis=0))
    y = torch.LongTensor(np.concatenate(ys, axis=0).astype(np.int64))
    if return_sid:
        return x, y, np.concatenate(sids, axis=0)
    return x, y

def fit_standardizer(subject_ids):
    """指定した学習被験者だけから、TUAB版と同じ axis=0 の mean/std を求める。"""
    x, _ = load_subjects(subject_ids)
    mean = x.mean(dim=0, keepdim=True)
    # np.std の既定(ddof=0)と合わせ、適用時にTUAB版と同じ1e-8を加える。
    std = x.std(dim=0, keepdim=True, correction=0)
    return mean, std


def make_loader(subject_ids, shuffle, batch_size=BATCH_SIZE, generator=None,
                train_mean=None, train_std=None, drop_last=False):
    x, y = load_subjects(subject_ids)

    # validation/test自身の統計量は使わず、対応するtrainの統計量だけを使用する。
    if train_mean is not None and train_std is not None:
        x = (x - train_mean) / (train_std + 1e-8)

    return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                      shuffle=shuffle, generator=generator,
                      drop_last=drop_last)


# =========================================================================
# Attention 関連
# =========================================================================
def cls_row_importance(attention_weights):
    """CLS行 attention から電極重要度を取る"""
    rows = []
    for layer in attention_weights:
        a = torch.stack(layer)                # [H, B, 18, 18] (CLS + 17ch)
        rows.append(a.mean(dim=0)[:, 0, 1:])  # ヘッド平均 -> CLS行 -> CLS列を除去
    return torch.stack(rows).mean(dim=0)      # 層平均 -> [B, 17]

# =========================================================================
# 学習・推論・評価関連の関数
# =========================================================================
@torch.no_grad()
def evaluate(model, loader, device, loss_fn, collect_attention=False):
    model.eval()
    total_loss, n_batch = 0.0, 0
    trues, preds, probs = [], [], []

    n_seen = 0
    sum_gat = torch.zeros(NUM_CHANNELS, NUM_CHANNELS)
    sum_col = torch.zeros(NUM_CHANNELS)
    sum_cls = torch.zeros(NUM_CHANNELS)

    for x, y in loader:
        if x.size(0) == 0:
            continue
        x, y = x.to(device), y.to(device).view(-1)
        
        # gcn_select_net は logits, attn_w, elec_attn, gat_attn を返す
        logits, attn_w, elec_attn, gat_attn = model(x)
        total_loss += loss_fn(logits, y).item()
        n_batch += 1

        pred = logits.argmax(dim=1)
        trues.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())

        if collect_attention:
            n_seen += x.size(0)
            # gat_attn: [B, heads, 17, 17] -> ヘッド平均
            sum_gat += gat_attn.mean(dim=1).sum(dim=0).detach().cpu()
            # elec_attn: [layers, B, 19] -> 層平均 -> CLS除去
            sum_col += elec_attn.mean(dim=0)[:, 1:].sum(dim=0).detach().cpu()
            sum_cls += cls_row_importance(attn_w).sum(dim=0).detach().cpu()

    labels = list(range(NUM_CLASSES))
    
    balanced_acc = balanced_accuracy_score(trues, preds)
    weighted_f1 = f1_score(trues, preds, average='weighted',
                           labels=labels, zero_division=0)
    kappa_val = cohen_kappa_score(trues, preds, labels=labels)
    if np.isnan(kappa_val):
        kappa_val = 0.0

    # Optuna内部では全指標を0--1に揃える。表示用accuracyだけ従来通り%にする。
    combined_score = 0.4 * balanced_acc + 0.3 * kappa_val + 0.3 * weighted_f1

    metrics = {
        'loss': total_loss / max(n_batch, 1),
        'acc': 100.0 * accuracy_score(trues, preds),
        'balanced_acc': float(balanced_acc),
        'balanced_acc_pct': 100.0 * float(balanced_acc),
        'macro_precision': precision_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_recall': recall_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_f1': f1_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'weighted_f1': float(weighted_f1),
        'kappa': float(kappa_val),
        'combined_score': float(combined_score),
        'n_class0_true': int(sum(1 for t in trues if t == 0)),
        'n_class1_true': int(sum(1 for t in trues if t == 1)),
        'both_classes_present': len(set(trues)) == NUM_CLASSES,
        'pred_fatigue_rate': float(np.mean(preds)) if preds else float('nan'),
    }
    
    attn = None
    if collect_attention and n_seen > 0:
        attn = {'gat': (sum_gat / n_seen).numpy(),
                'col': (sum_col / n_seen).numpy(),
                'cls': (sum_cls / n_seen).numpy()}
                
    return metrics, trues, preds, probs, attn

# =========================================================================
# 可視化
# =========================================================================
def plot_attention_matrix(matrix, tag, acc, out_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
                cmap="viridis", square=True)
    plt.title(f"{tag} GAT Attention (Target Acc: {acc:.2f}%)")
    plt.xlabel("Destination")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'attention_heatmap_{tag}.png'), dpi=150)
    plt.close()

    pos = np.array([COORDS[name] for name in CHANNEL_NAMES])
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.add_artist(plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2))
    plt.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k-', linewidth=2)
    for i, name in enumerate(CHANNEL_NAMES):
        x, y = pos[i]
        plt.scatter(x, y, s=500, c='white', edgecolors='black', zorder=10)
        plt.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold', zorder=11)

    threshold = np.percentile(matrix, 95)
    for i in range(len(CHANNEL_NAMES)):
        for j in range(len(CHANNEL_NAMES)):
            w = matrix[i, j]
            if w > threshold and i != j:
                p1, p2 = pos[i], pos[j]
                alpha = np.clip((w - threshold) / (matrix.max() - threshold + 1e-9), 0.1, 1.0)
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='red',
                         alpha=alpha, linewidth=3 * alpha, zorder=1)
    plt.title(f"{tag} Strongest Connections (Top 5%)")
    plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'attention_headmap_{tag}.png'), dpi=150)
    plt.close()

def plot_importance_bar(vector, tag, subtitle, out_dir, sort=False):
    names, vals = CHANNEL_NAMES, list(vector)
    if sort:
        order = np.argsort(vector)[::-1]
        names = [CHANNEL_NAMES[i] for i in order]
        vals = [vector[i] for i in order]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=vals, palette="Reds_r" if sort else "viridis")
    plt.title(f"{tag} Electrode Importance ({subtitle})")
    plt.xlabel("Channel"); plt.ylabel("Importance Score")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'electrode_importance_{tag}.png'), dpi=150)
    plt.close()

def plot_importance_head(vector, tag, out_dir):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.add_artist(plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2))
    plt.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k-', linewidth=2)
    vmin, vmax = float(np.min(vector)), float(np.max(vector))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Reds")
    for i, ch in enumerate(CHANNEL_NAMES):
        x, y = COORDS[ch]
        score = float(vector[i])
        size = 1000 * ((score - vmin) / (vmax - vmin + 1e-9)) + 200
        plt.scatter(x, y, s=size, c=[cmap(norm(score))], edgecolors='black', zorder=10)
        plt.text(x, y, ch, ha='center', va='center', fontsize=9, fontweight='bold', zorder=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Importance Score")
    plt.title(f"Spatial Distribution of Electrode Importance ({tag})")
    plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'electrode_importance_head_{tag}.png'), dpi=150)
    plt.close()

def plot_confusion(trues, preds, tag, title, out_dir):
    cm = confusion_matrix(trues, preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=CLASS_NAMES).plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'confusion_matrix_{tag}.png'), dpi=150)
    plt.close(fig)

# =========================================================================
# LOSO 実行ロジック
# =========================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_params(params=None):
    return {**DEFAULT_PARAMS, **(params or {})}


def suggest_params(trial):
    """TUAB版と同じ12項目を探索する。標準化方式は探索せずtrain基準に固定。"""
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_hidden_layers': trial.suggest_int('num_hidden_layers', 2, 6),
        'transformer_dropout': trial.suggest_float('transformer_dropout', 0.1, 0.5),
        'cnn_dropout': trial.suggest_float('cnn_dropout', 0.1, 0.5),
        'gnn_dropout': trial.suggest_float('gnn_dropout', 0.1, 0.5),
        'num_attention_heads': trial.suggest_categorical('num_attention_heads', [8, 16]),
        'gnn_heads': trial.suggest_categorical('gnn_heads', [1, 2, 4]),
        'cnn_out_channels': trial.suggest_categorical('cnn_out_channels', [8, 16, 32]),
        'use_lr_scheduler': trial.suggest_categorical('use_lr_scheduler', [True, False]),
    }


def build_model(params, device):
    params = merge_params(params)
    model_kwargs = {key: params[key] for key in MODEL_PARAM_KEYS}
    return gcn_select_net(num_classes=NUM_CLASSES, **model_kwargs).to(device)


def make_loss_fn(train_subjects, device):
    """inner trainだけのラベル頻度からTUAB方式のクラス重みを作る。"""
    _, y_train = load_subjects(train_subjects)
    y_np = y_train.numpy()
    present = np.unique(y_np)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    weights[present] = compute_class_weight(
        class_weight='balanced', classes=present, y=y_np).astype(np.float32)
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))


def train_model(train_subjects, epochs, device, seed, params, train_mean, train_std,
                val_subjects=None, patience=None, scheduler_t_max=None,
                trial=None, tag='', verbose=True):
    """1つのsubject-wise splitを学習する。

    validationを毎epoch評価し、TUAB版と同じ複合指標でbest checkpointを
    選ぶ。Optuna trial中は各epochのスコアを報告し、epoch単位でpruningする。
    """
    set_seed(seed)
    params = merge_params(params)

    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = make_loader(
        train_subjects, shuffle=True, batch_size=params['batch_size'],
        generator=gen, train_mean=train_mean, train_std=train_std,
        drop_last=True)
    val_loader = None
    if val_subjects:
        val_loader = make_loader(
            val_subjects, shuffle=False, batch_size=params['batch_size'],
            train_mean=train_mean, train_std=train_std)

    loss_fn = make_loss_fn(train_subjects, device)
    model = build_model(params, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params['learning_rate'],
        weight_decay=params['weight_decay'])
    scheduler = None
    if params['use_lr_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_t_max or epochs, eta_min=1e-6)

    curve_train_loss = []
    curve_val_score = []
    best_val_score = -float('inf')
    best_epoch = epochs
    best_model_state = None
    epochs_no_improve = 0

    for ep in range(epochs):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).view(-1)
            logits, _, _, _ = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)

        if scheduler is not None:
            scheduler.step()

        tr_loss = run_loss / max(len(train_loader), 1)
        epoch_acc = 100.0 * correct / seen
        curve_train_loss.append(tr_loss)

        val_text = ''
        if val_loader is not None:
            val_metrics, _, _, _, _ = evaluate(
                model, val_loader, device, loss_fn, collect_attention=False)
            val_score = val_metrics['combined_score']
            curve_val_score.append(val_score)
            val_text = (f" | val combined {val_score:.4f} "
                        f"(BACC {val_metrics['balanced_acc']:.4f}, "
                        f"Kappa {val_metrics['kappa']:.4f}, "
                        f"W-F1 {val_metrics['weighted_f1']:.4f})")
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = ep + 1
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if trial is not None:
                trial.report(val_score, step=ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if verbose:
            print(f"    {tag}Epoch {ep+1:3d}/{epochs} | train loss {tr_loss:.4f} "
                  f"acc {epoch_acc:5.2f}%{val_text}", flush=True)

        if (val_loader is not None and patience is not None
                and epochs_no_improve >= patience):
            if verbose:
                print(f"    {tag}Early stopping at epoch {ep + 1} "
                      f"(patience={patience})", flush=True)
            break

    # TUAB版と同様、後段に渡すモデルはvalidation最良checkpointに戻す。
    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'model': model,
        'loss_fn': loss_fn,
        'train_loss': curve_train_loss,
        'val_score': curve_val_score,
        'best_val_score': best_val_score if val_loader is not None else None,
        'best_epoch': best_epoch if val_loader is not None else epochs,
        'stopped_epoch': len(curve_train_loss),
    }


def make_inner_holdout(target, subjects, n_validation):
    """outer targetの次のセッションから循環的に、固定validation集合を作る。

    全outer foldを通じて各セッションがほぼ同じ回数validationに入り、分割はtrialや
    ハイパーパラメータに依存しない。23セッション・validation 3件なら各foldは
    inner train 19件 / validation 3件 / outer test 1件となる。
    """
    # 文字列の辞書順（1, 10, 11, ...）にはせず、SUBJECT_LIST由来の順序を保つ。
    ordered = list(dict.fromkeys(subjects))
    if target not in ordered:
        raise ValueError(f'target {target} がsubjectsに含まれていません')
    candidates = len(ordered) - 1
    if not 1 <= n_validation < candidates:
        raise ValueError(
            f'validationセッション数は1以上outer trainセッション数未満にしてください: '
            f'n_validation={n_validation}, outer_train={candidates}')

    target_index = ordered.index(target)
    rotated = ordered[target_index + 1:] + ordered[:target_index]
    validation_subjects = rotated[:n_validation]
    validation_set = set(validation_subjects)
    inner_train_subjects = [
        subject for subject in ordered
        if subject != target and subject not in validation_set
    ]
    return inner_train_subjects, validation_subjects


def make_objective(target, inner_train, validation_subjects,
                   inner_mean, inner_std, args, device):
    """outer targetを隔離した単一subject-wise HoldOutのOptuna目的関数。"""
    def objective(trial):
        params = merge_params(suggest_params(trial))
        try:
            # TUAB版と同じく、全trialを同じ学習seedで比較する。
            result = train_model(
                inner_train, args.max_epochs, device, SEED, params,
                inner_mean, inner_std, val_subjects=validation_subjects,
                patience=args.patience, scheduler_t_max=args.max_epochs,
                trial=trial, verbose=False, tag=f'[trial {trial.number}] ')
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f'CUDA OOM: {exc}') from exc
            raise

        score = float(result['best_val_score'])
        selected_epoch = int(result['best_epoch'])
        trial.set_user_attr('selected_epoch', selected_epoch)
        trial.set_user_attr('stopped_epoch', int(result['stopped_epoch']))
        trial.set_user_attr('validation_curve', result['val_score'])
        print(f"    Trial {trial.number:3d}: score={score:.4f}, "
              f"epoch={selected_epoch}, params={params}", flush=True)
        del result
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return score

    return objective


def optimize_fold(target, inner_train, validation_subjects, args, device,
                  save_path, inner_mean, inner_std):
    print(f"  [inner subject-wise HoldOut] {args.n_trials} trials")
    print(f"    train subjects      : {inner_train}")
    print(f"    validation subjects : {validation_subjects}")

    db_path = os.path.abspath(os.path.join(
        save_path, f'optuna_holdout_target_{target}.db'))
    study_name = f'seed_vig_outer_loso_inner_holdout_target_{target}_v3'
    sampler = optuna.samplers.TPESampler(seed=SEED + SUBJECT_INDEX[target])
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup,
        n_warmup_steps=args.pruner_warmup)
    study = optuna.create_study(
        direction='maximize', study_name=study_name,
        storage=f'sqlite:///{db_path}', load_if_exists=True,
        sampler=sampler, pruner=pruner)

    study_config = {
        'pipeline_version': 3,
        'dataset': 'SEED-VIG subject_wise_2class',
        'target': target,
        'inner_train_subjects': inner_train,
        'validation_subjects': validation_subjects,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'pruner_startup_trials': args.pruner_startup,
        'pruner_warmup_epochs': args.pruner_warmup,
        'standardization': 'train axis=0, population std',
        'objective': '0.4*bacc + 0.3*kappa + 0.3*weighted_f1',
    }
    previous_config = study.user_attrs.get('pipeline_config')
    if previous_config is not None and previous_config != study_config:
        raise RuntimeError(
            f'既存studyの設定が現在の設定と異なります: {db_path}\n'
            f'existing={previous_config}\ncurrent={study_config}\n'
            '別の--save-pathを指定するか、既存studyと同じ設定で再開してください。')
    study.set_user_attr('pipeline_config', study_config)

    finished_states = (optuna.trial.TrialState.COMPLETE,
                       optuna.trial.TrialState.PRUNED)
    n_finished = sum(t.state in finished_states for t in study.trials)
    remaining = max(0, args.n_trials - n_finished)
    if n_finished:
        print(f"    resume: finished={n_finished}, remaining={remaining}")
    if remaining:
        study.optimize(
            make_objective(
                target, inner_train, validation_subjects,
                inner_mean, inner_std, args, device),
            n_trials=remaining, timeout=args.optuna_timeout,
            gc_after_trial=True)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f'target {target}: completed Optuna trialがありません')

    study.trials_dataframe().to_csv(
        os.path.join(save_path, f'optuna_trials_target_{target}.csv'),
        index=False)
    best_trial = study.best_trial
    best = {
        'params': merge_params(best_trial.params),
        'epoch': int(best_trial.user_attrs['selected_epoch']),
        'score': float(best_trial.value),
        'trial': int(best_trial.number),
        'inner_train_subjects': inner_train,
        'validation_subjects': validation_subjects,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'objective': '0.4*balanced_accuracy + 0.3*kappa + 0.3*weighted_f1',
    }
    return best

def run_fold(target, kept, args, device, save_path):
    outer_train_subjects = [s for s in kept if s != target]
    inner_train, validation_subjects = make_inner_holdout(
        target, kept, args.val_subjects)
    inner_mean, inner_std = fit_standardizer(inner_train)

    print("=" * 70)
    print(f" OUTER LOSO + INNER HOLDOUT  ->  TARGET SUBJECT {target}")
    print("=" * 70)
    print(f"  inner train {len(inner_train)}セッション: {inner_train}")
    print(f"  validation  {len(validation_subjects)}セッション: {validation_subjects}")
    print(f"  outer test  1セッション: [{target}]", flush=True)

    if args.no_optuna:
        best = {
            'params': merge_params(), 'epoch': args.epochs,
            'score': None, 'trial': None,
            'inner_train_subjects': inner_train,
            'validation_subjects': validation_subjects,
            'max_epochs': args.epochs, 'patience': args.patience,
            'objective': '0.4*balanced_accuracy + 0.3*kappa + 0.3*weighted_f1',
        }
    else:
        best = optimize_fold(
            target, inner_train, validation_subjects, args, device,
            save_path, inner_mean, inner_std)

    # Optuna選択内容を先に保存し、最終再学習後に実測best epoch等で更新する。
    with open(os.path.join(save_path, f'inner_target_{target}.json'), 'w') as f:
        json.dump({'target': target, **best}, f, indent=2,
                  ensure_ascii=False)

    # TUAB版と同様、同じinner train/validationで最初から再学習し、
    # validation複合スコアが最大のcheckpointをouter testへ適用する。
    np.savez(os.path.join(save_path, f'standardizer_target_{target}.npz'),
             mean=inner_mean.numpy(), std=inner_std.numpy(),
             train_subjects=np.asarray(inner_train, dtype=str),
             validation_subjects=np.asarray(
                 validation_subjects, dtype=str),
             target=np.asarray([target], dtype=str))

    print(f"  [final train] train={len(inner_train)}, "
          f"validation={len(validation_subjects)}, "
          f"max_epochs={best['max_epochs']}, params={best['params']}",
          flush=True)
    final = train_model(
        inner_train, best['max_epochs'], device, SEED,
        best['params'], inner_mean, inner_std,
        val_subjects=validation_subjects, patience=args.patience,
        scheduler_t_max=best['max_epochs'], tag=f'[{target}] ')
    model, loss_fn = final['model'], final['loss_fn']
    best['optuna_selected_epoch'] = (
        None if args.no_optuna else best['epoch'])
    best['epoch'] = int(final['best_epoch'])
    best['final_validation_score'] = float(final['best_val_score'])
    best['final_stopped_epoch'] = int(final['stopped_epoch'])

    with open(os.path.join(save_path, f'inner_target_{target}.json'), 'w') as f:
        json.dump({'target': target, **best}, f, indent=2,
                  ensure_ascii=False)

    torch.save(model.state_dict(),
               os.path.join(save_path, f'model_target_{target}.pth'))
    return eval_target(
        model, target, best, inner_train, validation_subjects,
        outer_train_subjects, loss_fn, final['train_loss'],
        final['val_score'], args, device, save_path,
        train_mean=inner_mean, train_std=inner_std)

def eval_target(model, target, best, train_subjects, validation_subjects,
                outer_train_subjects, loss_fn, curve_train_loss,
                curve_val_score, args, device, save_path,
                train_mean=None, train_std=None):
    
    # ターゲットの評価時にも、Trainデータの統計量を用いて標準化する
    params = merge_params(best['params'])
    test_loader = make_loader(
        [target], shuffle=False, batch_size=params['batch_size'],
        train_mean=train_mean, train_std=train_std)

    # AdaBNなしで評価 (Attention情報も取得)
    tm, trues, preds, probs, attn = evaluate(model, test_loader, device, loss_fn, collect_attention=True)

    print(f"\n  [Target {target}] test内訳 覚醒{tm['n_class0_true']} / 疲労{tm['n_class1_true']}"
          f"{'' if tm['both_classes_present'] else '  ※単一クラスのため指標が退化'}")
    print(f"  真の疲労率 {tm['n_class1_true'] / max(len(trues), 1):.3f} / "
          f"予測疲労率 {tm['pred_fatigue_rate']:.3f}")
    print(f"  Accuracy {tm['acc']:.2f}% | BACC {tm['balanced_acc_pct']:.2f}% "
          f"| Precision {tm['macro_precision']:.4f} | Recall {tm['macro_recall']:.4f} "
          f"| F1-Score {tm['macro_f1']:.4f} | Kappa {tm['kappa']:.4f}\n",
          flush=True)

    result = {**tm, 'target': target, 'n_test': len(trues),
              'train_subjects': train_subjects,
              'validation_subjects': validation_subjects,
              'outer_train_subjects': outer_train_subjects,
              'epochs': best['epoch'], 'params': params,
              'optuna_selected_epoch': best.get('optuna_selected_epoch'),
              'final_validation_score': best.get('final_validation_score'),
              'inner_score': best['score'], 'optuna_trial': best['trial'],
              'selection_max_epochs': best['max_epochs'],
              'early_stop_patience': best['patience'],
              'optuna_objective': best['objective'],
              'standardization': 'inner train axis=0, population std'}

    with open(os.path.join(save_path, f'fold_target_{target}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(save_path, f'fold_target_{target}.npz'),
             trues=np.array(trues), preds=np.array(preds),
             probs=np.array(probs, dtype=np.float32),
             gat=attn['gat'], col=attn['col'], cls=attn['cls'],
             curve_train_loss=np.array(curve_train_loss),
             curve_val_score=np.array(curve_val_score))

    tag = f'target_{target}'
    plot_attention_matrix(attn['gat'], tag, tm['acc'], save_path)
    plot_importance_bar(attn['cls'], tag, f"CLS-row, Acc {tm['acc']:.2f}%", save_path)
    plot_confusion(trues, preds, tag,
                   f'Target Subject {target} (Acc: {tm["acc"]:.2f}%)', save_path)
    return result

# =========================================================================
# 集約
# =========================================================================
def print_prior_bias_diagnosis(fold_results, targets, arrays):
    if len(targets) < 3:
        return
    true_rate, pred_rate, accs, recs = [], [], [], []
    for s in targets:
        r = fold_results[s]
        true_rate.append(r['n_class1_true'] / max(r['n_test'], 1))
        pr = r.get('pred_fatigue_rate')
        pred_rate.append(float(arrays[s]['preds'].mean()) if pr is None else pr)
        accs.append(r['acc'])
        recs.append(r['macro_recall'])

    true_rate, pred_rate = np.array(true_rate), np.array(pred_rate)
    corr_acc = float(np.corrcoef(true_rate, accs)[0, 1])
    corr_rec = float(np.corrcoef(true_rate, recs)[0, 1])

    print("\n===== 事前分布バイアスの診断 =====")
    print(f"  真の疲労率 平均 {true_rate.mean():.3f} / 予測疲労率 平均 {pred_rate.mean():.3f} "
          f"(差 {pred_rate.mean() - true_rate.mean():+.3f})")
    print(f"  真の疲労率 vs Accuracy          の相関: r = {corr_acc:+.3f}")
    print(f"  真の疲労率 vs Recall(=balanced) の相関: r = {corr_rec:+.3f}")
    if corr_acc < -0.4 and abs(corr_rec) < 0.3:
        print("  -> 疲労が多い被験者で識別能力が落ちているのではなく、出力が"
              f"{'覚醒' if pred_rate.mean() < true_rate.mean() else '疲労'}寄りに偏っているために")
        print("     Accuracy だけが押し下げられている。被験者間の比較には Recall を使うこと。")
    elif corr_rec < -0.4:
        print("  -> Recall とも負相関。疲労が多い被験者で実際に識別できていない。")
    return {'corr_acc': corr_acc, 'corr_recall': corr_rec,
            'mean_true_fatigue_rate': float(true_rate.mean()),
            'mean_pred_fatigue_rate': float(pred_rate.mean())}


def aggregate(save_path, manifest):
    kept = manifest['kept_subjects']
    balanced = manifest['loso_target_subjects']

    files = sorted(glob.glob(os.path.join(save_path, 'fold_target_*.json')))
    if not files:
        print(f"集約対象がありません: {save_path}")
        return
    fold_results, arrays = {}, {}
    for p in files:
        r = json.load(open(p))
        t = r['target']
        fold_results[t] = r
        arrays[t] = np.load(p.replace('.json', '.npz'))
    targets = sorted(
        fold_results,
        key=lambda subject: SUBJECT_INDEX.get(subject, len(SUBJECT_LIST)))
    degenerate = [s for s in targets if s not in balanced]

    global_true = np.concatenate([arrays[t]['trues'] for t in targets]).tolist()
    global_pred = np.concatenate([arrays[t]['preds'] for t in targets]).tolist()

    def summarize(subset):
        sub = [fold_results[s] for s in subset if s in fold_results]
        if not sub:
            return None
        out = {'n_subjects': len(sub), 'subjects': list(subset)}
        for key, _ in REPORT_METRICS:
            v = np.array([r[key] for r in sub], dtype=float)
            v = v[~np.isnan(v)]
            out[key] = {'mean': float(v.mean()), 'std': float(v.std())} if len(v) else None
        return out

    summary_all = summarize(targets)
    summary_balanced = summarize([s for s in targets if s in balanced])

    report = classification_report(
        global_true, global_pred, labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES, digits=4, zero_division=0)
    
    pooled = {
        'accuracy': accuracy_score(global_true, global_pred),
        'balanced_accuracy': balanced_accuracy_score(global_true, global_pred),
        'macro_precision': precision_score(global_true, global_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(global_true, global_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(global_true, global_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(global_true, global_pred, average='weighted', zero_division=0),
        'kappa': float(cohen_kappa_score(global_true, global_pred)),
    }
    pooled['combined_score'] = (0.4 * pooled['balanced_accuracy']
                                + 0.3 * pooled['kappa']
                                + 0.3 * pooled['weighted_f1'])

    def print_summary(title, summ):
        if summ is None:
            return
        print(f"\n {title} (n={summ['n_subjects']}セッション)")
        for key, label in REPORT_METRICS:
            v = summ[key]
            if v is None:
                continue
            pct = key in PCT_METRICS
            unit = '%' if pct else ''
            fmt = '.2f' if pct else '.4f'
            print(f"   {label} : {v['mean']:{fmt}}{unit} ± {v['std']:{fmt}}{unit}")

    print("\n" + "=" * 70)
    selected_epochs = {s: int(fold_results[s].get('epochs', 0)) for s in targets}
    validation_subjects_per_fold = {
        s: fold_results[s].get('validation_subjects', []) for s in targets
    }
    print(f" OUTER LOSO + INNER HOLDOUT RESULTS ({len(targets)}/{len(kept)} folds)")
    print(f" selected epochs {selected_epochs}")
    print("=" * 70)
    print_summary('【全ターゲット】', summary_all)
    if degenerate:
        print_summary('【両クラスを持つセッションのみ】', summary_balanced)

    print("\n セッション別:")
    for s in targets:
        r = fold_results[s]
        mark = '' if s in balanced else '  ※退化'
        true_rate = r['n_class1_true'] / max(r['n_test'], 1)
        pred_rate = r.get('pred_fatigue_rate')
        if pred_rate is None:
            pred_rate = float(arrays[s]['preds'].mean())
        print(f"  {s:<20s} n={r['n_test']:<5d} (覚醒{r['n_class0_true']:4d}/疲労{r['n_class1_true']:4d})"
              f"  疲労率 真{true_rate:.3f}/予{pred_rate:.3f}"
              f"  Acc {r['acc']:6.2f}%  BACC {r['balanced_acc_pct']:6.2f}%"
              f"  Prec {r['macro_precision']:.4f}"
              f"  Rec {r['macro_recall']:.4f}  F1 {r['macro_f1']:.4f}  Kappa {r['kappa']:.4f}{mark}")

    print_prior_bias_diagnosis(fold_results, targets, arrays)
    print("\n===== Pooled Classification Report =====")
    print(report)

    plot_confusion(global_true, global_pred, 'global',
                   f'LOSO Global (Acc: {pooled["accuracy"]*100:.2f}%, '
                   f'F1: {pooled["macro_f1"]:.4f}, Kappa: {pooled["kappa"]:.4f})', save_path)

    # --- 電極重要度の集約 ---
    imp_col = np.stack([arrays[t]['col'] for t in targets])
    imp_cls = np.stack([arrays[t]['cls'] for t in targets])
    gat_mats = np.stack([arrays[t]['gat'] for t in targets])
    np.save(os.path.join(save_path, 'importance_colmean_per_fold.npy'), imp_col)
    np.save(os.path.join(save_path, 'importance_clsrow_per_fold.npy'), imp_cls)
    np.save(os.path.join(save_path, 'gat_attention_per_fold.npy'), gat_mats)
    np.save(os.path.join(save_path, 'fold_order.npy'), np.array(targets))

    ranking_txt = []
    for name, arr in [('clsrow', imp_cls), ('colmean', imp_col)]:
        mean_imp = arr.mean(axis=0)
        order = np.argsort(mean_imp)[::-1]
        plot_importance_bar(mean_imp, f'global_{name}',
                            f'mean of {len(arr)} LOSO folds', save_path, sort=True)
        plot_importance_head(mean_imp, f'global_{name}', save_path)

        tops = [set(np.argsort(-a)[:8]) for a in arr]
        cnt = np.zeros(NUM_CHANNELS, dtype=int)
        for t in tops:
            for i in t:
                cnt[i] += 1
        ranking_txt.append(f"===== Global Electrode Importance ({name}) =====")
        for rank, i in enumerate(order, 1):
            ranking_txt.append(f"{rank:2d}. {CHANNEL_NAMES[i]:4s} {mean_imp[i]:.6f}"
                               f"   (top-8入り {cnt[i]}/{len(arr)} folds)")
        ranking_txt.append("")
        print(f"\n===== Electrode Importance ({name}) =====")
        print("  " + ", ".join(CHANNEL_NAMES[i] for i in order))

    plot_attention_matrix(gat_mats.mean(axis=0), 'global', pooled['accuracy'] * 100, save_path)

    # --- 学習曲線（train loss のみ） ---
    try:
        n_ep = min(len(arrays[t]['curve_train_loss']) for t in targets)
        tl = np.stack([arrays[t]['curve_train_loss'][:n_ep] for t in targets])
        ep_axis = np.arange(1, n_ep + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ep_axis, tl.mean(axis=0), label='Train Loss', color='tab:blue')
        ax.fill_between(ep_axis, tl.mean(axis=0) - tl.std(axis=0),
                        tl.mean(axis=0) + tl.std(axis=0), color='tab:blue', alpha=0.2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Train Loss')
        ax.set_title(f'LOSO average learning curve ({len(targets)} folds, first {n_ep} epochs)')
        ax.legend(); fig.tight_layout()
        plt.savefig(os.path.join(save_path, 'learning_curve.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plotting Error: {e}")

    # --- 保存 ---
    with open(os.path.join(save_path, 'loso_results.json'), 'w') as f:
        json.dump({
            'data_dir': DATA_DIR,
            'labeling_rule': manifest['labeling_rule'],
            'excluded_subjects': manifest['excluded_subjects'],
            'kept_subjects': kept,
            'target_subjects': targets,
            'degenerate_targets': degenerate,
            'selected_epochs': selected_epochs,
            'validation_subjects_per_fold': validation_subjects_per_fold,
            'selection_pipeline': 'outer LOSO + inner subject-wise HoldOut with Optuna',
            'inner_validation_subjects': (
                len(next(iter(validation_subjects_per_fold.values())))
                if validation_subjects_per_fold else 0),
            'standardization': 'inner train axis=0, population std',
            'per_subject': {str(s): fold_results[s] for s in targets},
            'summary_all_targets': summary_all,
            'summary_balanced_targets': summary_balanced,
            'pooled': pooled,
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_path, 'loso_report.txt'), 'w') as f:
        f.write("===== LOSO 2-class (outer LOSO + inner subject-wise HoldOut + Optuna) =====\n")
        f.write("metrics: Accuracy / Balanced Accuracy / Precision / Recall / F1-Score / Kappa "
                "(Precision, Recall, F1 は macro 平均)\n")
        f.write(f"selected epochs per fold: {selected_epochs}\n")
        f.write(f"validation subjects per fold: {validation_subjects_per_fold}\n")
        f.write("standardization: train-only axis=0 mean/population std\n")
        f.write("Optuna objective: 0.4*balanced accuracy + 0.3*kappa "
                "+ 0.3*weighted F1\n")
        f.write(f"labeling: {manifest['labeling_rule']}\n")
        f.write(f"target subjects  : {targets}\n")
        f.write("\n")
        for title, summ in [('全ターゲット', summary_all),
                            ('両クラスを持つセッションのみ', summary_balanced)]:
            if summ is None or (title != '全ターゲット' and not degenerate):
                continue
            f.write(f"----- {title} (n={summ['n_subjects']}) -----\n")
            for key, label in REPORT_METRICS:
                v = summ[key]
                if v is None:
                    continue
                pct = key in PCT_METRICS
                unit = '%' if pct else ''
                fmt = '.2f' if pct else '.4f'
                f.write(f"  {label}: {v['mean']:{fmt}}{unit} ± {v['std']:{fmt}}{unit}\n")
            f.write("\n")
            
        f.write("Per subject:\n")
        for s in targets:
            r = fold_results[s]
            mark = '' if s in balanced else '  ※退化'
            f.write(f"  {s}: n={r['n_test']} (awake={r['n_class0_true']}/"
                    f"fatigue={r['n_class1_true']}) acc={r['acc']:.2f}% "
                    f"bacc={r['balanced_acc_pct']:.2f}% "
                    f"prec={r['macro_precision']:.4f} rec={r['macro_recall']:.4f} "
                    f"f1={r['macro_f1']:.4f} kappa={r['kappa']:.4f}{mark}\n")

        f.write("\n===== Pooled Classification Report =====\n")
        f.write(report + "\n")
        f.write("\n".join(ranking_txt))

    print(f"\nSaved to {save_path}")


# =========================================================================
# multi-GPU outer-fold scheduler
# =========================================================================
def query_gpu_memory():
    """nvidia-smiから物理GPUごとの空き/総メモリ(MiB)を取得する。"""
    command = [
        'nvidia-smi',
        '--query-gpu=index,memory.free,memory.total',
        '--format=csv,noheader,nounits',
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            'nvidia-smiが見つからないためmulti-GPU監視を開始できません。') from exc

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f'nvidia-smiによるGPUメモリ取得に失敗しました: {message}')

    stats = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 3:
            raise RuntimeError(f'nvidia-smiの予期しない出力です: {line!r}')
        try:
            gpu_id, free_mb, total_mb = map(int, fields)
        except ValueError as exc:
            raise RuntimeError(
                f'nvidia-smiの出力を数値に変換できません: {line!r}') from exc
        stats[gpu_id] = {'free_mb': free_mb, 'total_mb': total_mb}

    if not stats:
        raise RuntimeError('利用可能なGPUをnvidia-smiから取得できませんでした。')
    return stats


def select_scheduler_gpu_ids(gpu_spec, available_ids):
    """--gpu-idsを解釈する。auto時は数値CUDA_VISIBLE_DEVICESも尊重する。"""
    available = set(available_ids)
    if gpu_spec == 'auto':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is not None:
            tokens = [token.strip() for token in visible.split(',') if token.strip()]
            if tokens and all(token.isdigit() for token in tokens):
                selected = [int(token) for token in tokens]
            elif not tokens:
                raise RuntimeError('CUDA_VISIBLE_DEVICESが空なので利用可能なGPUがありません。')
            else:
                raise RuntimeError(
                    'CUDA_VISIBLE_DEVICESがGPU UUID/MIG形式です。'
                    '--gpu-idsに物理GPU indexを明示してください。')
        else:
            selected = sorted(available)
    else:
        try:
            selected = [int(token.strip()) for token in gpu_spec.split(',')
                        if token.strip()]
        except ValueError as exc:
            raise RuntimeError(
                '--gpu-idsは auto または 0,1,2 のように指定してください。') from exc

    # 重複指定は順序を維持して除去する。
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError('--gpu-idsでGPUが1つも選択されていません。')
    missing = [gpu_id for gpu_id in selected if gpu_id not in available]
    if missing:
        raise RuntimeError(
            f'指定GPU {missing} はnvidia-smiの一覧 {sorted(available)} にありません。')
    return selected


def make_worker_command(args, target):
    """親schedulerと同じ実験条件を単一fold workerへ渡す。"""
    command = [
        sys.executable, '-u', os.path.abspath(__file__),
        '--only-target', str(target),
        '--epochs', str(args.epochs),
        '--max-epochs', str(args.max_epochs),
        '--n-trials', str(args.n_trials),
        '--val-subjects', str(args.val_subjects),
        '--patience', str(args.patience),
        '--pruner-startup', str(args.pruner_startup),
        '--pruner-warmup', str(args.pruner_warmup),
        '--save-path', os.path.abspath(args.save_path),
        '--scheduler-worker',
    ]
    if args.optuna_timeout is not None:
        command.extend(['--optuna-timeout', str(args.optuna_timeout)])
    if args.no_optuna:
        command.append('--no-optuna')
    return command


def format_gpu_status(stats, gpu_ids, running):
    fields = []
    for gpu_id in gpu_ids:
        states = [state for state in running.values()
                  if state['gpu_id'] == gpu_id]
        task = ('idle' if not states else 'targets=' + ','.join(
            str(state['target']) for state in states))
        fields.append(
            f"GPU{gpu_id}: free={stats[gpu_id]['free_mb']}/"
            f"{stats[gpu_id]['total_mb']} MiB, {task} ({len(states)} workers)")
    return ' | '.join(fields)


def run_multi_gpu_scheduler(targets, args, manifest):
    """1回の起動で、outer foldを空きGPUへ動的に割り当てる。"""
    initial_stats = query_gpu_memory()
    gpu_ids = select_scheduler_gpu_ids(args.gpu_ids, initial_stats)
    log_dir = (os.path.abspath(args.worker_log_dir) if args.worker_log_dir
               else os.path.join(os.path.abspath(args.save_path), 'worker_logs'))
    os.makedirs(log_dir, exist_ok=True)

    print('\n===== MULTI-GPU OUTER-FOLD SCHEDULER =====')
    print(f'GPU IDs        : {gpu_ids}')
    print(f'memory/worker  : {args.min_free_memory_mb} MiB (scheduler予約値)')
    print(f'memory reserve : {args.gpu_memory_reserve_mb} MiB / GPU')
    print(f'workers/GPU    : max {args.max_workers_per_gpu}')
    print(f'poll interval  : {args.gpu_poll_seconds:.1f} sec')
    print(f'worker retries : {args.gpu_retries}')
    print(f'worker logs    : {log_dir}')
    print(format_gpu_status(initial_stats, gpu_ids, {}), flush=True)

    pending = list(targets)
    running = {}
    completed_targets = []
    failed_targets = {}
    attempts = {target: 0 for target in targets}
    last_status_time = 0.0
    memory_wait_started = None

    try:
        while pending or running:
            # 終了workerを先に回収し、そのGPUを次の割当に利用可能にする。
            for worker_id, state in list(running.items()):
                return_code = state['process'].poll()
                if return_code is None:
                    continue
                state['log_handle'].close()
                target = state['target']
                gpu_id = state['gpu_id']
                del running[worker_id]
                result_exists = os.path.exists(state['result_path'])
                if return_code == 0 and result_exists:
                    completed_targets.append(target)
                    print(f'[scheduler] target {target} completed on GPU{gpu_id}',
                          flush=True)
                elif attempts[target] <= args.gpu_retries:
                    pending.append(target)
                    print(
                        f'[scheduler] target {target} failed '
                        f'(exit={return_code}, result_exists={result_exists}); '
                        f'retry {attempts[target]}/{args.gpu_retries}を待機列へ戻します。'
                        f" log={state['log_path']}", flush=True)
                else:
                    failed_targets[target] = {
                        'return_code': return_code,
                        'result_exists': result_exists,
                        'log': state['log_path'],
                    }
                    print(
                        f'[scheduler] target {target} failed permanently '
                        f"(exit={return_code}, log={state['log_path']})", flush=True)

            if not pending and not running:
                break

            try:
                stats = query_gpu_memory()
            except RuntimeError as exc:
                # 監視不能時は安全側に倒し、新しいworkerを割り当てない。
                print(f'[scheduler] GPU監視エラー。新規割当を保留します: {exc}',
                      flush=True)
                time.sleep(args.gpu_poll_seconds)
                continue

            assigned = False
            for gpu_id in gpu_ids:
                active_count = sum(
                    state['gpu_id'] == gpu_id for state in running.values())
                # 同一poll内に複数workerを起動しても、CUDA確保がnvidia-smiへ
                # 反映される前に過剰割当しないよう、起動ごとに仮想的に差し引く。
                schedulable_free_mb = stats[gpu_id]['free_mb']
                while (pending and active_count < args.max_workers_per_gpu
                       and schedulable_free_mb >= (
                           args.min_free_memory_mb
                           + args.gpu_memory_reserve_mb)):
                    target = pending.pop(0)
                    attempts[target] += 1
                    attempt = attempts[target]
                    log_path = os.path.join(
                        log_dir,
                        f'target_{target}_gpu_{gpu_id}_attempt_{attempt}.log')
                    log_handle = open(log_path, 'a', buffering=1)
                    env = os.environ.copy()
                    # worker内からは割り当てた物理GPUがcuda:0として見える。
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                    env['PYTHONUNBUFFERED'] = '1'
                    command = make_worker_command(args, target)
                    try:
                        process = subprocess.Popen(
                            command, stdout=log_handle,
                            stderr=subprocess.STDOUT, env=env,
                            cwd=os.path.dirname(os.path.dirname(
                                os.path.abspath(__file__))))
                    except BaseException:
                        log_handle.close()
                        pending.insert(0, target)
                        raise
                    running[process.pid] = {
                        'process': process,
                        'gpu_id': gpu_id,
                        'target': target,
                        'log_handle': log_handle,
                        'log_path': log_path,
                        'result_path': os.path.join(
                            os.path.abspath(args.save_path),
                            f'fold_target_{target}.json'),
                        'assigned_free_mb': schedulable_free_mb,
                    }
                    schedulable_free_mb -= args.min_free_memory_mb
                    active_count += 1
                    assigned = True
                    print(
                        f'[scheduler] target {target} -> GPU{gpu_id} '
                        f'(observed free={stats[gpu_id]["free_mb"]} MiB, '
                        f'reserved remainder={schedulable_free_mb} MiB, '
                        f'worker={active_count}/{args.max_workers_per_gpu}, '
                        f'pid={process.pid}, log={log_path})', flush=True)

            now = time.monotonic()
            if now - last_status_time >= args.gpu_status_seconds:
                print('[GPU monitor] ' + format_gpu_status(
                    stats, gpu_ids, running), flush=True)
                if pending:
                    print(f'[scheduler] pending targets: {pending}', flush=True)
                last_status_time = now

            if pending and not running and not assigned:
                if memory_wait_started is None:
                    memory_wait_started = now
                if (args.gpu_wait_timeout > 0
                        and now - memory_wait_started >= args.gpu_wait_timeout):
                    raise RuntimeError(
                        f'{args.gpu_wait_timeout}秒待機しても空きメモリ '
                        f'{args.min_free_memory_mb} MiB以上のGPUがありません。')
            else:
                memory_wait_started = None

            if pending or running:
                time.sleep(args.gpu_poll_seconds)
    except BaseException:
        # 親が中断された場合に孤立workerを残さない。
        for state in running.values():
            process = state['process']
            if process.poll() is None:
                process.terminate()
            state['log_handle'].close()
        raise

    if failed_targets:
        details = ', '.join(
            f"target {target}: {info['log']}"
            for target, info in sorted(failed_targets.items()))
        raise RuntimeError(
            f'{len(failed_targets)} foldが再試行後も失敗しました。'
            f'不完全な結果は集約しません。{details}')

    print(f'[scheduler] all {len(completed_targets)} folds completed.',
          flush=True)
    aggregate(args.save_path, manifest)


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=FIXED_EPOCHS,
                    help=f'--no-optuna時の固定エポック数（既定 {FIXED_EPOCHS}）')
    ap.add_argument('--max-epochs', type=int, default=FIXED_EPOCHS,
                    help=f'各Optuna trialの最大エポック数（既定 {FIXED_EPOCHS}）')
    ap.add_argument('--n-trials', type=int, default=N_TRIALS,
                    help=f'outer foldごとの総trial数（既定 {N_TRIALS}、DB再開分を含む）')
    ap.add_argument('--val-subjects', type=int, default=INNER_VAL_SUBJECTS,
                    help=f'各outer foldの固定validationセッション数（既定 {INNER_VAL_SUBJECTS}）')
    ap.add_argument('--patience', type=int, default=EARLY_STOP_PATIENCE,
                    help=f'inner学習のearly stopping patience（既定 {EARLY_STOP_PATIENCE}）')
    ap.add_argument('--pruner-startup', type=int, default=5,
                    help='MedianPrunerが枝刈りを始める前の完走trial数')
    ap.add_argument('--pruner-warmup', type=int, default=PRUNER_WARMUP_EPOCHS,
                    help=f'pruningを行わない先頭epoch数（既定 {PRUNER_WARMUP_EPOCHS}）')
    ap.add_argument('--optuna-timeout', type=int, default=None,
                    help='outer foldごとのOptuna時間上限（秒、既定は制限なし）')
    ap.add_argument('--no-optuna', action='store_true',
                    help='Optunaを行わず既定パラメータと--epochsで実行する')
    ap.add_argument('--limit-folds', type=int, default=None,
                    help='先頭Nセッションだけ回す（動作確認用）')
    ap.add_argument(
        '--targets', choices=['all', 'balanced'], default='all',
        help='all: 配置済み全セッション（既定） / balanced: 両クラスを十分持つセッション')
    ap.add_argument('--only-target', type=str, default=None,
                    help='このセッション1件のfoldだけ実行する（並列実行用）。集約は行わない。')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='学習せず、保存済みのfold結果から集約だけ行う')
    ap.add_argument('--multi-gpu', action='store_true',
                    help='1回の起動でouter foldを複数GPUへ動的に割り当てる')
    ap.add_argument('--gpu-ids', default='auto',
                    help='multi-GPUで使う物理GPU index（例: 0,1,2）。既定auto')
    ap.add_argument('--min-free-memory-mb', type=int,
                    default=MIN_FREE_GPU_MEMORY_MB,
                    help='worker 1つ分として予約するGPUメモリMiB')
    ap.add_argument('--gpu-memory-reserve-mb', type=int,
                    default=GPU_MEMORY_RESERVE_MB,
                    help='workerへ割り当てずGPUに残す安全余裕MiB')
    ap.add_argument('--max-workers-per-gpu', type=int,
                    default=MAX_WORKERS_PER_GPU,
                    help='1 GPUで同時実行するouter fold数の上限（既定3）')
    ap.add_argument('--gpu-poll-seconds', type=float,
                    default=GPU_POLL_SECONDS,
                    help='GPU空き容量とworker終了を確認する間隔（秒）')
    ap.add_argument('--gpu-status-seconds', type=float,
                    default=GPU_STATUS_SECONDS,
                    help='GPU状態を親ログへ表示する間隔（秒）')
    ap.add_argument('--gpu-wait-timeout', type=float, default=0,
                    help='全GPUメモリ不足時の最大待機秒数（0は無制限）')
    ap.add_argument('--gpu-retries', type=int, default=1,
                    help='異常終了したfoldの再試行回数（既定1）')
    ap.add_argument('--worker-log-dir', default=None,
                    help='multi-GPU workerログ保存先（既定: save-path/worker_logs）')
    ap.add_argument('--scheduler-worker', action='store_true',
                    help=argparse.SUPPRESS)
    ap.add_argument('--save-path', type=str, default=SAVE_PATH)
    
    args = ap.parse_args()

    if args.epochs < 1 or args.max_epochs < 1:
        ap.error('--epochs と --max-epochs は1以上にしてください')
    if args.n_trials < 1 and not args.no_optuna:
        ap.error('--n-trials は1以上にしてください')
    if args.val_subjects < 1:
        ap.error('--val-subjects は1以上にしてください')
    if args.patience < 1:
        ap.error('--patience は1以上にしてください')
    if args.pruner_warmup < 0:
        ap.error('--pruner-warmup は0以上にしてください')
    if args.min_free_memory_mb < 1:
        ap.error('--min-free-memory-mbは1以上にしてください')
    if args.gpu_memory_reserve_mb < 0:
        ap.error('--gpu-memory-reserve-mbは0以上にしてください')
    if args.max_workers_per_gpu < 1:
        ap.error('--max-workers-per-gpuは1以上にしてください')
    if args.gpu_poll_seconds <= 0 or args.gpu_status_seconds <= 0:
        ap.error('--gpu-poll-secondsと--gpu-status-secondsは0より大きくしてください')
    if args.gpu_wait_timeout < 0:
        ap.error('--gpu-wait-timeoutは0以上にしてください')
    if args.gpu_retries < 0:
        ap.error('--gpu-retriesは0以上にしてください')
    if args.multi_gpu and args.only_target is not None:
        ap.error('--multi-gpuと--only-targetは同時に指定できません')
    if args.multi_gpu and args.aggregate_only:
        ap.error('--multi-gpuと--aggregate-onlyは同時に指定できません')
    if args.scheduler_worker and args.only_target is None:
        ap.error('--scheduler-workerには--only-targetが必要です')

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    manifest = build_manifest()
    kept = manifest['kept_subjects']
    balanced = manifest['loso_target_subjects']
    if not kept:
        raise SystemExit(
            f'SEED-VIGデータが見つかりません: {DATA_DIR}\n'
            'eeg_<session>.npy と label_<session>.npy を配置してください。')
    if args.val_subjects >= len(kept) - 1:
        ap.error('--val-subjects はouter trainセッション数未満にしてください')

    if args.only_target is not None and args.only_target not in kept:
        ap.error(f'--only-target {args.only_target} はkept_subjectsに含まれません')

    # 親プロセスまたは通常実行だけが共有manifestを書き、multi-GPU worker間の
    # 同時上書きを避ける。
    if not args.scheduler_worker:
        with open(os.path.join(save_path, 'manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.aggregate_only:
        aggregate(save_path, manifest)
        return

    if args.only_target is not None:
        targets = [args.only_target]
    else:
        targets = kept if args.targets == 'all' else balanced
        if args.limit_folds:
            targets = targets[:args.limit_folds]

    split_manifest = {}
    for target in targets:
        inner_train, validation_subjects = make_inner_holdout(
            target, kept, args.val_subjects)
        split_manifest[str(target)] = {
            'inner_train_subjects': inner_train,
            'validation_subjects': validation_subjects,
            'outer_test_subject': target,
        }
    # multi-GPU worker同士が同じmanifestを上書きしないよう、単一fold実行では
    # target固有名にする。親schedulerは全fold版holdout_splits.jsonを保存する。
    split_filename = ('holdout_splits.json' if args.only_target is None
                      else f'holdout_split_target_{args.only_target}.json')
    with open(os.path.join(save_path, split_filename), 'w') as f:
        json.dump({
            'all_subjects': kept,
            'targets': targets,
            'n_validation_subjects': args.val_subjects,
            'assignment': 'cyclic next sessions in canonical SUBJECT_LIST order',
            'folds': split_manifest,
        }, f, indent=2, ensure_ascii=False)

    print(f"データ  : {DATA_DIR}")
    print("モデル  : gcn_select_net (Attention可視化あり)")
    print(f"学習可  : {len(kept)}セッション {kept}")
    print(f"ターゲット: {len(targets)}セッション {targets}")
    if args.no_optuna:
        print(f"学習設定: Optunaなし (epochs={args.epochs}, params={merge_params()})")
    else:
        print(f"学習設定: outer LOSO + inner subject-wise HoldOut, "
              f"Optuna {args.n_trials} trials, validation={args.val_subjects}セッション, "
              f"max_epochs={args.max_epochs}, patience={args.patience}")
        print(f"枝刈り  : MedianPruner(startup={args.pruner_startup}, "
              f"warmup={args.pruner_warmup} epochs)")
    print("標準化  : inner trainのみのaxis=0 mean・population std")
    print("目的関数: 0.4*BACC + 0.3*Kappa + 0.3*weighted F1")
    print(f"保存先  : {save_path}\n", flush=True)

    if args.multi_gpu:
        run_multi_gpu_scheduler(targets, args, manifest)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.scheduler_worker and device.type != 'cuda':
        raise RuntimeError(
            'multi-GPU workerでCUDAを利用できません。割当GPUとCUDA環境を確認してください。')
    set_seed(SEED)
    print(f"Using device: {device}", flush=True)

    for target in targets:
        run_fold(target, kept, args, device, save_path)

    if args.only_target is None:
        aggregate(save_path, manifest)
    else:
        print(f"fold (target={args.only_target}) 完了。")

if __name__ == '__main__':
    main()
