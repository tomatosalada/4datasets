"""
SEED-VIG データを用いた 2クラス(覚醒/疲労) の LOSO 学習ベースライン実験。
追加処理（入れ子CV, AdaBN, SWA, 平滑化, クラス重み）をすべて撤廃した純粋なベースラインです。

【前処理の変更】
データリークを防ぐため、各LOSOのfoldにおいて、Trainデータ全体から
平均(mean)と標準偏差(std)を計算し、それをTrain/Testの両方に適用する
グローバル標準化方式を採用しています。
"""

import argparse
import copy
import glob
import json
import os
import sys
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score,
                             cohen_kappa_score)
from torch.utils.data import DataLoader, TensorDataset

# プロジェクトルートへのパスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# モデルとして gcn_select_net をimport (SEED-VIG用パス)
from select_channel.select_net_class import gcn_select_net

warnings.filterwarnings("ignore")

# =========================================================================
# 設定
# =========================================================================
DATA_DIR = '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/processedData/subject_wise_2class'
SAVE_PATH = '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/LOSO_pretrainedmodel/LOSO_GCN_Baseline/'

NUM_CLASSES = 2
CLASS_NAMES = ['Awake', 'Fatigue']

# SEED-VIG の 'chn' の並びから CPZ を除いた17ch
CHANNEL_NAMES = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
                 'CP1', 'CP2', 'P1', 'PZ', 'P2',
                 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']
NUM_CHANNELS = len(CHANNEL_NAMES)

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

BATCH_SIZE = 128
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.03
SEED = 42

FIXED_EPOCHS = 100

REPORT_METRICS = [('acc', 'Accuracy '),
                  ('macro_precision', 'Precision'),
                  ('macro_recall', 'Recall   '),
                  ('macro_f1', 'F1-Score ')]
PCT_METRICS = {'acc'}

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

def load_subjects(subject_ids, return_sid=False):
    """被験者ごとの個別標準化や平滑化を行わず、生データをそのまま結合する"""
    xs, ys, sids = [], [], []
    for s in subject_ids:
        x = np.load(os.path.join(DATA_DIR, f'eeg_{s}.npy')).astype(np.float32)
        y = np.load(os.path.join(DATA_DIR, f'label_{s}.npy'))
            
        xs.append(x)
        ys.append(y)
        sids.append(np.full(len(y), SUBJECT_INDEX[s], dtype=np.int64))
        
    x = torch.FloatTensor(np.concatenate(xs, axis=0))
    y = torch.LongTensor(np.concatenate(ys, axis=0).astype(np.int64))
    if return_sid:
        return x, y, np.concatenate(sids, axis=0)
    return x, y

def make_loader(subject_ids, shuffle, generator=None, train_mean=None, train_std=None):
    x, y = load_subjects(subject_ids)
    
    # --- グローバル標準化 (データリーク防止) ---
    if train_mean is not None and train_std is not None:
        x = (x - train_mean) / (train_std + 1e-8)
    # ----------------------------------------------------------------------
    
    # 通常のランダムサンプリングを使用
    return DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)

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
            # elec_attn: [layers, B, 18] -> 層平均 -> CLS除去
            sum_col += elec_attn.mean(dim=0)[:, 1:].sum(dim=0).detach().cpu()
            sum_cls += cls_row_importance(attn_w).sum(dim=0).detach().cpu()

    labels = list(range(NUM_CLASSES))
    
    metrics = {
        'loss': total_loss / max(n_batch, 1),
        'acc': 100.0 * accuracy_score(trues, preds),
        'balanced_acc': 100.0 * balanced_accuracy_score(trues, preds),
        'macro_precision': precision_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_recall': recall_score(trues, preds, average='macro', labels=labels, zero_division=0),
        'macro_f1': f1_score(trues, preds, average='macro', labels=labels, zero_division=0),
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
def train_model(train_subjects, epochs, device, seed, lr, tag='', verbose=True, train_mean=None, train_std=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    gen = torch.Generator()
    gen.manual_seed(seed)
    
    # train_mean と train_std を Loader に渡す
    train_loader = make_loader(train_subjects, shuffle=True, generator=gen, train_mean=train_mean, train_std=train_std)
    loss_fn = nn.CrossEntropyLoss()
    
    model = gcn_select_net(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    curve_train_loss = []
    
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

        tr_loss = run_loss / len(train_loader)
        epoch_acc = 100.0 * correct / seen
        curve_train_loss.append(tr_loss)
        
        # 最高精度を上回ったら重みを保存
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"    {tag}Epoch {ep+1:3d}/{epochs} | train loss {tr_loss:.4f} "
                  f"acc {epoch_acc:5.2f}%", flush=True)

    # ベストの重みを復元
    model.load_state_dict(best_model_wts)
    if verbose:
        print(f"    {tag}Best Train Acc: {best_acc:5.2f}% loaded for evaluation.", flush=True)

    return model, loss_fn, curve_train_loss

def run_fold(target, kept, balanced, args, device, save_path):
    train_subjects = [s for s in kept if s != target]

    print("=" * 70)
    print(f" LOSO  ->  TARGET SUBJECT {target}")
    print("=" * 70)
    print(f"  train {len(train_subjects)}セッション")
    print(f"  test  1セッション: [{target}]", flush=True)

    # --- Trainデータ全体から mean / std を計算 (データリーク防止) ---
    x_train_raw, _ = load_subjects(train_subjects)
    train_mean = x_train_raw.mean(dim=0, keepdim=True)
    train_std = x_train_raw.std(dim=0, keepdim=True)
    del x_train_raw
    # -------------------------------------------------------------------

    print(f"  [train] {len(train_subjects)}セッションで学習: lr={LEARNING_RATE:g} epochs={args.epochs}", flush=True)
          
    model, loss_fn, curve_train_loss = train_model(
        train_subjects, args.epochs, device, SEED + SUBJECT_INDEX[target],
        LEARNING_RATE, tag=f'[{target}] ', train_mean=train_mean, train_std=train_std)

    torch.save(model.state_dict(), os.path.join(save_path, f'model_target_{target}.pth'))
    return eval_target(model, target, train_subjects, loss_fn,
                       curve_train_loss, args, device, save_path, train_mean=train_mean, train_std=train_std)

def eval_target(model, target, train_subjects, loss_fn,
                curve_train_loss, args, device, save_path, train_mean=None, train_std=None):
    
    # ターゲットの評価時にも Train データの統計量を用いて標準化
    test_loader = make_loader([target], shuffle=False, train_mean=train_mean, train_std=train_std)

    tm, trues, preds, probs, attn = evaluate(model, test_loader, device, loss_fn, collect_attention=True)

    print(f"\n  [Target {target}] test内訳 覚醒{tm['n_class0_true']} / 疲労{tm['n_class1_true']}"
          f"{'' if tm['both_classes_present'] else '  ※単一クラスのため指標が退化'}")
    print(f"  真の疲労率 {tm['n_class1_true'] / max(len(trues), 1):.3f} / "
          f"予測疲労率 {tm['pred_fatigue_rate']:.3f}")
    print(f"  Accuracy {tm['acc']:.2f}% | Precision {tm['macro_precision']:.4f} "
          f"| Recall {tm['macro_recall']:.4f} | F1-Score {tm['macro_f1']:.4f}\n", flush=True)

    result = {**tm, 'target': target, 'n_test': len(trues),
              'train_subjects': train_subjects,
              'epochs': args.epochs, 'lr': LEARNING_RATE}

    with open(os.path.join(save_path, f'fold_target_{target}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(save_path, f'fold_target_{target}.npz'),
             trues=np.array(trues), preds=np.array(preds),
             probs=np.array(probs, dtype=np.float32),
             gat=attn['gat'], col=attn['col'], cls=attn['cls'],
             curve_train_loss=np.array(curve_train_loss))

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
        
    targets = sorted(fold_results, key=lambda s: SUBJECT_INDEX.get(s, len(SUBJECT_LIST)))
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

    report = classification_report(global_true, global_pred,
                                   target_names=CLASS_NAMES, digits=4, zero_division=0)
    
    pooled = {
        'accuracy': accuracy_score(global_true, global_pred),
        'macro_precision': precision_score(global_true, global_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(global_true, global_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(global_true, global_pred, average='macro', zero_division=0),
    }

    def print_summary(title, summ):
        if summ is None:
            return
        print(f"\n {title} (n={summ['n_subjects']}名)")
        for key, label in REPORT_METRICS:
            v = summ[key]
            if v is None:
                continue
            pct = key in PCT_METRICS
            unit = '%' if pct else ''
            fmt = '.2f' if pct else '.4f'
            print(f"   {label} : {v['mean']:{fmt}}{unit} ± {v['std']:{fmt}}{unit}")

    print("\n" + "=" * 70)
    print(f" LOSO RESULTS (Baseline) ({len(targets)}/{len(kept)} folds)")
    print(f" epochs {fold_results[targets[0]].get('epochs')}")
    print("=" * 70)
    print_summary('【全ターゲット】', summary_all)
    if degenerate:
        print_summary('【両クラスを持つ被験者のみ】', summary_balanced)

    print("\n 被験者別:")
    for s in targets:
        r = fold_results[s]
        mark = '' if s in balanced else '  ※退化'
        true_rate = r['n_class1_true'] / max(r['n_test'], 1)
        pred_rate = r.get('pred_fatigue_rate')
        if pred_rate is None:
            pred_rate = float(arrays[s]['preds'].mean())
        print(f"  {s:<20s} n={r['n_test']:<5d} (覚醒{r['n_class0_true']:4d}/疲労{r['n_class1_true']:4d})"
              f"  疲労率 真{true_rate:.3f}/予{pred_rate:.3f}"
              f"  Acc {r['acc']:6.2f}%  Prec {r['macro_precision']:.4f}"
              f"  Rec {r['macro_recall']:.4f}  F1 {r['macro_f1']:.4f}{mark}")

    print_prior_bias_diagnosis(fold_results, targets, arrays)
    print("\n===== Pooled Classification Report =====")
    print(report)

    plot_confusion(global_true, global_pred, 'global',
                   f'LOSO Global (Acc: {pooled["accuracy"]*100:.2f}%, '
                   f'F1: {pooled["macro_f1"]:.4f})', save_path)

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
        ax.set_title(f'LOSO average learning curve ({len(targets)} folds, {n_ep} epochs fixed)')
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
            'fixed_epochs': fold_results[targets[0]].get('epochs'),
            'per_subject': {str(s): fold_results[s] for s in targets},
            'summary_all_targets': summary_all,
            'summary_balanced_targets': summary_balanced,
            'pooled': pooled,
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(save_path, 'loso_report.txt'), 'w') as f:
        f.write("===== LOSO 2-class (SEED-VIG subject_wise_2class Baseline) =====\n")
        f.write("metrics: Accuracy / Precision / Recall / F1-Score "
                "(Precision, Recall, F1 は macro 平均)\n")
        f.write(f"epochs : {fold_results[targets[0]].get('epochs')} (全fold固定)\n")
        f.write(f"labeling: {manifest['labeling_rule']}\n")
        f.write(f"target subjects  : {targets}\n")
        f.write("\n")
        for title, summ in [('全ターゲット', summary_all),
                            ('両クラスを持つ被験者のみ', summary_balanced)]:
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
                    f"prec={r['macro_precision']:.4f} rec={r['macro_recall']:.4f} "
                    f"f1={r['macro_f1']:.4f}{mark}\n")

        f.write("\n===== Pooled Classification Report =====\n")
        f.write(report + "\n")
        f.write("\n".join(ranking_txt))

    print(f"\nSaved to {save_path}")

# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=FIXED_EPOCHS,
                    help=f'学習エポック数。全fold共通（既定 {FIXED_EPOCHS}）')
    ap.add_argument('--limit-folds', type=int, default=None,
                    help='先頭N名だけ回す（動作確認用）')
    ap.add_argument('--targets', choices=['all', 'balanced'], default='all')
    ap.add_argument('--only-target', type=str, default=None,
                    help='このセッション1件のfoldだけ実行する（並列実行用）。集約は行わない。')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='学習せず、保存済みのfold結果から集約だけ行う')
    ap.add_argument('--save-path', type=str, default=SAVE_PATH)
    
    args = ap.parse_args()

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    manifest = build_manifest()
    kept = manifest['kept_subjects']
    balanced = manifest['loso_target_subjects']

    if not kept:
        raise SystemExit(f"データが見つかりません: {DATA_DIR}\n"
                         f"先にデータを配置してください")
    
    with open(os.path.join(save_path, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.aggregate_only:
        aggregate(save_path, manifest)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.only_target is not None:
        targets = [args.only_target]
    else:
        targets = kept if args.targets == 'all' else balanced
        if args.limit_folds:
            targets = targets[:args.limit_folds]

    degenerate = [s for s in targets if s not in balanced]
    print(f"データ  : {DATA_DIR}")
    print(f"モデル  : gcn_select_net (Attention可視化あり)")
    print(f"学習可  : {len(kept)}セッション")
    print(f"ターゲット: {len(targets)}セッション")
    print(f"学習設定: Baseline (epochs {args.epochs} / lr {LEARNING_RATE:g} / グローバル標準化)")
    print(f"保存先  : {save_path}\n", flush=True)

    for target in targets:
        run_fold(target, kept, balanced, args, device, save_path)

    if args.only_target is None:
        aggregate(save_path, manifest)
    else:
        print(f"fold (target={args.only_target}) 完了。")

if __name__ == '__main__':
    main()