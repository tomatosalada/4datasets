# =========================================================================
# 提案パターン vs 「てんかんで生理学的に重視される部位」候補 の比較実験
#
#   目的: C(16,4) の全探索ではなく、神経生理学的に根拠のある少数の候補と
#         提案パターン (事前学習の重要度 top-K) を、複数シードで比較する。
#
#   本スクリプトは全探索 (combo_fixelectrode_forpaper.py) から完全に独立している。
#   all_combos.csv は参照しない。出力も candidate_results_* 配下で完結する。
#   候補数が少ないぶん 1候補あたり複数シードを回せるので、点推定ではなく
#   平均±SD と対応ありの検定で比較できる。
#
#   記録するもの (候補×シードごとに CSV 1行):
#     - BACC / F1(macro) / Kappa / Accuracy   ← 主指標は3本立て (BACC, F1, Kappa)
#     - クラス別 recall (TUAB: Normal / Abnormal)
#       ※ 固定部位ごとに正常/異常の recall バランス (異常検出感度と正常特異度の
#         トレードオフ) がどう変わるかを検証するため。
#
#   使い方:
#     python candidate_benchmark_forpaper_fix3.py                    # 既定 total=11, fixed=3, 5 seeds
#     python candidate_benchmark_forpaper_fix3.py --seeds 42 43 44 45 46
#     python candidate_benchmark_forpaper_fix3.py --procs-per-gpu 2  # 全探索と併走させる場合は下げる
#     python candidate_benchmark_forpaper_fix3.py --analyze-only     # 学習せず集計だけやり直す
# =========================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import time
import argparse
import traceback
import signal
import faulthandler


def _enable_fault_diagnostics():
    """死因・ハング原因をログに残せるようにする。

    faulthandler.enable()      : SIGSEGV/SIGBUS/SIGFPE で落ちた場合に C レベルの
                                 スタックを stderr に吐く (= 沈黙して消えるのを防ぐ)。
    faulthandler.register(...) : 生きたまま固まった場合、外から
                                   kill -USR1 <PID>
                                 を送るとその時点のPythonスタックを吐く。
                                 「学習中なのか、デッドロックなのか」を判別できる。
    """
    faulthandler.enable()
    if hasattr(signal, 'SIGUSR1'):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


_enable_fault_diagnostics()

import torch
import torch.multiprocessing as mp
import numpy as np
import warnings
from sklearn.metrics import recall_score

from ratio_fixdynamic_forpaper_eeggate import (
    SEED, BATCH_SIZE, DEFAULT_EPOCH,
    LR_GATE, LR_CLASSIFIER, LAMBDA_SPARSITY,
    PRETRAINED_MODEL_PATH,
    CHANNEL_NAMES, NUM_CHANNELS, IMPORTANCE_RANKING, CLASS_LABELS,
    PROCESSES_PER_GPU,
    load_data, train_eval_holdout,
)

warnings.filterwarnings("ignore")

DEFAULT_TOTAL_ELECTRODES = 11
DEFAULT_FIXED_ELECTRODES = 3
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

OUTPUT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CH_INDEX = {name: i for i, name in enumerate(CHANNEL_NAMES)}

# =========================================================================
# 比較候補: てんかんの神経生理学・臨床所見に基づく固定電極パターン
#
#   16ch は双極 TCP モンタージュ:
#     左側頭鎖   FP1-F7, F7-T3, T3-T5, T5-O1
#     右側頭鎖   FP2-F8, F8-T4, T4-T6, T6-O2
#     左傍矢状鎖 FP1-F3, F3-C3, C3-P3, P3-O1
#     右傍矢状鎖 FP2-F4, F4-C4, C4-P4, P4-O2
#
#   rationale はそのまま論文の Table 説明文に流用できる粒度で書いてある。
#
#   ★ 固定数 3 版について:
#     16ch は左右対称な 4 鎖で構成されるため、4ch 候補は左右対称に組める。
#     3ch では対称性を保てないので、各候補から「その領域の rationale への寄与が
#     最も小さい 1ch」を外して 3ch にしている (左右対称ペアを核として残し、
#     外した 1ch は rationale 内に明記)。結果として候補ごとに左寄り/右寄りが
#     生じるため、側方性が交絡し得る点は Limitation として記載すること。
# =========================================================================
CANDIDATES = {
    'proposed': dict(
        channels=None,  # None = IMPORTANCE_RANKING[:FIXED] を実行時に解決 (提案手法)
        group='proposed',
        rationale='事前学習済みモデルのチャネル重要度 上位K (提案手法)'),

    'anterior_temporal': dict(
        channels=['F7-T3', 'F8-T4', 'T4-T6'],
        group='clinical',
        rationale='側頭葉てんかんは成人の焦点性てんかんで最多。内側側頭起源の発作間欠期'
                  'てんかん性放電は前〜中側頭 (F7/T3, F8/T4) で振幅最大となる。'
                  '3本構成のため左中側頭 T3-T5 を外し、振幅最大の前側頭両側 (F7-T3, F8-T4) を'
                  '核として右中側頭 T4-T6 を残す。'),

    'frontotemporal': dict(
        channels=['FP1-F7', 'FP2-F8', 'F7-T3'],
        group='clinical',
        rationale='前頭葉てんかんは焦点性てんかんで側頭葉に次いで多い。前頭下面〜側頭前部を'
                  'カバーし、焦点性発作の大半の起始領域を含む。'
                  '3本構成のため、前頭下面の左右対称ペア (FP1-F7, FP2-F8) を核として残し、'
                  '右深部側頭 F8-T4 を外して左前側頭 F7-T3 を残す。'),

    'centrotemporal': dict(
        channels=['C3-P3', 'C4-P4', 'T4-T6'],
        group='clinical',
        rationale='中心側頭部棘波を伴う自然終息性小児てんかん (ローランドてんかん) の'
                  '棘波分布。中心-側頭移行部に最大電位をもつ。'
                  '3本構成のため、中心-頭頂の左右対称ペア (C3-P3, C4-P4) を核として残し、'
                  '左後側頭 T3-T5 を外して右側頭 T4-T6 を残す。'),

    'frontopolar': dict(
        channels=['FP1-F7', 'FP2-F8', 'FP1-F3'],
        group='clinical',
        rationale='前頭極。全般性周期性放電・三相波は前頭優位で、前頭部の異常所見を捉えやすい。'
                  '前頭ウェアラブル EEG の実運用形態にも対応。'
                  '3本構成のため、前頭極外側の左右対称ペア (FP1-F7, FP2-F8) を核として残し、'
                  '右内側 FP2-F4 を外して左内側 FP1-F3 を残す。'),

    'occipital': dict(
        channels=['P3-O1', 'P4-O2', 'T6-O2'],
        group='clinical',
        rationale='後頭葉てんかん (Panayiotopoulos 型・Gastaut 型) の放電分布。'
                  '後頭部および側頭後部をカバー。'
                  '3本構成のため、頭頂-後頭の左右対称ペア (P3-O1, P4-O2) を核として残し、'
                  '左後側頭 T5-O1 を外して右後側頭 T6-O2 を残す。'),

    'parasagittal': dict(
        channels=['C3-P3', 'C4-P4', 'F3-C3'],
        group='clinical',
        rationale='傍矢状・中心領域。全般性放電が捉えやすく、側頭筋由来の筋電アーチファクトが'
                  '最も少ない領域でもある (ARTF クラスとの関係を検証できる)。'
                  '3本構成のため、中心-頭頂の左右対称ペア (C3-P3, C4-P4) を核として残し、'
                  '右前中心 F4-C4 を外して左前中心 F3-C3 を残す。'),

    'symmetric_pairs': dict(
        channels=['F7-T3', 'F8-T4', 'C3-P3'],
        group='clinical',
        rationale='前方 (側頭) の左右相同ペア (F7-T3, F8-T4) を核として残し、後方は C3-P3 の'
                  '1本のみ。対称ペアは片側性周期性放電 (PLED) の側方性判定に必要な最小構成だが、'
                  '3本構成では後方 (中心-頭頂) の対称性が崩れ C4-P4 を外すため、後方の'
                  '側方性判定能が前方より弱くなる点は Limitation に記す。'),

    'left_temporal': dict(
        channels=['FP1-F7', 'F7-T3', 'T3-T5'],
        group='control',
        rationale='左側頭鎖の前方3本のみ (後頭寄りの T5-O1 を外す)。側方性が一方に偏った'
                  '場合の対照 (PLED 等の片側性事象に有利、対側事象に不利になるはず)。'),

    'right_temporal': dict(
        channels=['FP2-F8', 'F8-T4', 'T4-T6'],
        group='control',
        rationale='右側頭鎖の前方3本のみ (後頭寄りの T6-O2 を外す)。left_temporal との対比で'
                  '左右非対称性の影響を見る。'),
}

CLASS_ORDER = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]

FIELDNAMES = (
    ['candidate', 'group', 'seed', 'combo_id', 'fixed_indices', 'fixed_channels',
     'n_total_electrodes', 'n_fixed_electrodes', 'n_variable_target',
     'avg_selected_electrodes', 'val_acc',
     'eval_acc', 'eval_bacc', 'eval_f1', 'eval_kappa',
     'eval_precision', 'eval_recall']
    + [f'recall_{c}' for c in CLASS_ORDER]
    + [f'f1_{c}' for c in CLASS_ORDER]
    + ['status']
)


def combo_key(indices):
    return "-".join(f"{i:02d}" for i in sorted(indices))


def resolve_candidates(fixed_n):
    """CANDIDATES を (name -> インデックス list) に解決する。
    channels=None の 'proposed' は事前学習ランキング上位 fixed_n で埋める。
    電極数が fixed_n と一致しない候補はスキップし、理由を表示する。"""
    out = {}
    for name, spec in CANDIDATES.items():
        chs = spec['channels']
        if chs is None:
            chs = list(IMPORTANCE_RANKING[:fixed_n])
        if len(chs) != fixed_n:
            print(f"  [skip] {name}: 電極数 {len(chs)} != --fixed {fixed_n}")
            continue
        unknown = [c for c in chs if c not in CH_INDEX]
        if unknown:
            raise ValueError(f"{name}: 未知のチャネル名 {unknown}")
        out[name] = dict(spec, channels=chs,
                         indices=sorted(CH_INDEX[c] for c in chs))
    return out


# =========================================================================
# ワーカー
# =========================================================================
_G = {}


def _init_worker(data_tuple, num_classes, total, fixed, epoch, num_gpus, verbose_train):
    _enable_fault_diagnostics()  # spawn では継承されないので子でも有効化
    _G.update(data=data_tuple, num_classes=num_classes, total=total,
              fixed=fixed, epoch=epoch, num_gpus=num_gpus,
              verbose_train=verbose_train)
    print(f"  [worker pid={os.getpid()}] initialized "
          f"(X_train={tuple(data_tuple[0].shape)})", flush=True)


def _set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _base_row(name, spec, seed, total, fixed):
    return {
        'candidate': name,
        'group': spec['group'],
        'seed': seed,
        'combo_id': combo_key(spec['indices']),
        'fixed_indices': ';'.join(map(str, spec['indices'])),
        'fixed_channels': ';'.join(spec['channels']),
        'n_total_electrodes': total,
        'n_fixed_electrodes': fixed,
        'n_variable_target': max(0, total - fixed),
    }


def run_one(task):
    """(候補, シード) を1本学習・評価して CSV 1行分の dict を返す。"""
    worker_idx, name, spec, seed = task

    num_gpus = _G['num_gpus']
    gpu_id = worker_idx % num_gpus if num_gpus > 0 else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # 開始時にも1行出す。これが出て結果行が出ないなら「学習中に落ちた」、
    # これすら出ないなら「プール起動〜タスク配布で落ちた」と切り分けられる。
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(gpu_id)
        mem = f"GPU{gpu_id} free={free_b/2**30:.1f}/{total_b/2**30:.1f}GiB"
    else:
        mem = "cpu"
    print(f"  -> START {name}/seed{seed} (pid={os.getpid()}, {mem})", flush=True)

    _set_seed(seed)  # ★ シードのみが候補間で共通の乱数条件を作る

    X_train, y_train, X_val, y_val, X_eval, y_eval = _G['data']
    num_classes = _G['num_classes']
    total, fixed, epoch = _G['total'], _G['fixed'], _G['epoch']

    fixed_indices = list(spec['indices'])
    variable_indices = [i for i in range(NUM_CHANNELS) if i not in fixed_indices]
    target_var = max(0, total - fixed)
    run_label = f"{name}/seed{seed}"

    row = _base_row(name, spec, seed, total, fixed)
    try:
        r = train_eval_holdout(
            X_train, y_train, X_val, y_val, X_eval, y_eval,
            num_classes, NUM_CHANNELS, fixed_indices, variable_indices,
            BATCH_SIZE, epoch, target_var, run_label,
            LAMBDA_SPARSITY, LR_GATE, LR_CLASSIFIER,
            PRETRAINED_MODEL_PATH, "", device,
            save_model=False, verbose=_G['verbose_train'],
        )
        # クラス別 recall / f1 (部位とクラスの対応を検証するため)
        from sklearn.metrics import f1_score as _f1
        per_rec = recall_score(r['eval_true'], r['eval_pred'],
                               labels=list(range(num_classes)),
                               average=None, zero_division=0)
        per_f1 = _f1(r['eval_true'], r['eval_pred'],
                     labels=list(range(num_classes)), average=None, zero_division=0)

        row.update({
            'avg_selected_electrodes': round(float(r['val_elec']), 4),
            'val_acc': round(float(r['val_acc']), 4),
            'eval_acc': round(float(r['eval_acc']), 4),
            'eval_bacc': round(float(r['eval_bacc']), 6),
            'eval_f1': round(float(r['eval_f1']), 6),
            'eval_kappa': round(float(r['eval_kap']), 6),
            'eval_precision': round(float(r['eval_prec']), 6),
            'eval_recall': round(float(r['eval_rec']), 6),
            'status': 'OK',
        })
        for i, c in enumerate(CLASS_ORDER):
            row[f'recall_{c}'] = round(float(per_rec[i]), 6)
            row[f'f1_{c}'] = round(float(per_f1[i]), 6)
        return row

    except Exception as e:
        print(f"[{run_label}] FAILED: {e}", flush=True)
        traceback.print_exc()
        for k in FIELDNAMES:
            row.setdefault(k, float('nan'))
        row['status'] = f'FAILED: {e}'
        return row


# =========================================================================
# 集計
# =========================================================================
def analyze(csv_path, res_dir, fixed_n):
    import pandas as pd
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'OK'].drop_duplicates(
        subset=['candidate', 'seed'], keep='last')
    if df.empty:
        print("No OK rows yet. Skip analysis.")
        return

    METRICS = ['eval_bacc', 'eval_f1', 'eval_kappa']   # 主指標3本
    LABELS = {'eval_bacc': 'Balanced Accuracy', 'eval_f1': 'Macro F1',
              'eval_kappa': "Cohen's Kappa", 'eval_acc': 'Accuracy'}

    agg = df.groupby(['candidate', 'group'])[METRICS + ['eval_acc']] \
            .agg(['mean', 'std', 'count'])
    agg = agg.reset_index()

    lines = []
    add = lines.append
    add("===== Candidate benchmark: proposed vs. clinically-motivated fixed sets =====")
    add(f"seeds per candidate: {sorted(df['seed'].unique())}")
    add("")

    # --- 主指標3本の表 (mean ± sd) ---
    hdr = f"{'candidate':<20} {'group':<9}" + "".join(f"{LABELS[m]:>24}" for m in METRICS)
    add(hdr)
    add("-" * len(hdr))
    order = agg.sort_values(('eval_bacc', 'mean'), ascending=False)
    for _, r in order.iterrows():
        s = f"{r[('candidate', '')]:<20} {r[('group', '')]:<9}"
        for m in METRICS:
            s += f"{r[(m,'mean')]:>15.4f} ±{r[(m,'std')]:.4f}" if r[(m, 'count')] > 1 \
                 else f"{r[(m,'mean')]:>15.4f}  (n=1) "
        add(s)
    add("")

    # --- 提案 vs 各候補: シード対応ありの検定 ---
    add("--- Paired comparison against 'proposed' (same seeds, paired t-test) ---")
    add("  positive delta = the candidate BEATS the proposed set")
    prop = df[df['candidate'] == 'proposed'].set_index('seed')
    for m in METRICS:
        add(f"  [{LABELS[m]}]")
        for name in df['candidate'].unique():
            if name == 'proposed':
                continue
            other = df[df['candidate'] == name].set_index('seed')
            common = sorted(set(prop.index) & set(other.index))
            if len(common) < 2:
                add(f"    {name:<20} (seed 数不足で検定不可, n={len(common)})")
                continue
            a, b = other.loc[common, m].values, prop.loc[common, m].values
            d = a - b
            t, p = stats.ttest_rel(a, b)
            sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            add(f"    {name:<20} delta={d.mean():+.4f} ±{d.std(ddof=1):.4f} "
                f"(n={len(common)}, p={p:.3f}){sig}")
        add("")

    # --- クラス別 recall (部位 -> クラスの対応を検証) ---
    add("--- Per-class recall (mean over seeds) ---")
    rec_cols = [f'recall_{c}' for c in CLASS_ORDER]
    rec = df.groupby('candidate')[rec_cols].mean()
    rec = rec.loc[order[('candidate', '')].values]
    add(f"{'candidate':<20}" + "".join(f"{c:>9}" for c in CLASS_ORDER))
    for name, r in rec.iterrows():
        add(f"{name:<20}" + "".join(f"{r[f'recall_{c}']:>9.3f}" for c in CLASS_ORDER))
    add("  ※ TUAB は Normal/Abnormal の2クラス。固定部位ごとに正常/異常の recall バランスが")
    add("     どう変わるか (異常検出感度と正常特異度のトレードオフ) を確認する。")
    add("")

    # --- 指標ごとの候補内順位 (指標によって勝者が入れ替わるかを一目で見る) ---
    add("--- Rank among candidates, per metric (1 = best) ---")
    add("  指標間で順位が入れ替わる場合、単一指標での「勝者」主張はできない。")
    ranks = {}
    for m in METRICS:
        means = agg.set_index(('candidate', ''))[(m, 'mean')]
        ranks[m] = means.rank(ascending=False).astype(int)
    add(f"{'candidate':<20}" + "".join(f"{LABELS[m]:>20}" for m in METRICS))
    for _, r in order.iterrows():
        name = r[('candidate', '')]
        s = f"{name:<20}"
        for m in METRICS:
            cell = f"#{ranks[m][name]} ({r[(m, 'mean')]:.4f})"
            s += f"{cell:>20}"
        add(s)
    add("")

    txt = "\n".join(lines)
    with open(os.path.join(res_dir, 'candidate_summary.txt'), 'w') as f:
        f.write(txt + "\n")
    agg.to_csv(os.path.join(res_dir, 'candidate_aggregate.csv'))
    print("\n" + txt)

    # =====================================================================
    # 図1: 主指標3本 (mean±sd, 候補ごと) — 提案を強調
    # =====================================================================
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False})
    COLORS = {'proposed': '#c0504d', 'clinical': '#3b6fb6', 'control': '#9aa4b1'}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.4))
    for ax, m in zip(np.atleast_1d(axes), METRICS):
        sub = agg.sort_values((m, 'mean'), ascending=True)
        names = sub[('candidate', '')].values
        means = sub[(m, 'mean')].values
        sds = np.nan_to_num(sub[(m, 'std')].values)
        cols = [COLORS[g] for g in sub[('group', '')].values]
        y = np.arange(len(names))
        ax.barh(y, means, xerr=sds, color=cols, alpha=.9,
                error_kw=dict(lw=1, capsize=2.5, ecolor='#444'))
        # 個々のシードを点で重ねる (分散が見えるように)
        for yi, nm in zip(y, names):
            vals = df.loc[df['candidate'] == nm, m].values
            ax.scatter(vals, np.full_like(vals, yi, dtype=float), s=8,
                       color='black', alpha=.45, zorder=3)
        ax.set_yticks(y, names)
        lo = max(0, means.min() - 4 * (sds.max() + 1e-3))
        ax.set_xlim(lo, means.max() + 3 * (sds.max() + 1e-3))
        ax.set_xlabel(LABELS[m])
        ax.set_title(LABELS[m])
    fig.suptitle('Proposed vs. clinically-motivated fixed electrode sets '
                 '(mean ± sd over seeds; dots = individual seeds)', y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, 'fig_candidates_metrics.png'))
    plt.close(fig)

    # =====================================================================
    # 図2: クラス別 recall ヒートマップ (部位 -> クラスの対応)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(1.1 * len(CLASS_ORDER) + 3.2,
                                    0.42 * len(rec) + 1.8))
    mat = rec.values
    im = ax.imshow(mat, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_yticks(range(len(rec)), rec.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center',
                    fontsize=7,
                    color='white' if mat[i, j] > mat.max() * .75 else 'black')
    ax.set_title('Per-class recall by fixed electrode set (mean over seeds)')
    fig.colorbar(im, ax=ax, shrink=.8, label='recall')
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, 'fig_candidates_class_recall.png'))
    plt.close(fig)

    print(f"\nWrote to {res_dir}/:")
    for fn in ['candidate_summary.txt', 'candidate_aggregate.csv',
               'fig_candidates_metrics.png', 'fig_candidates_class_recall.png']:
        p = os.path.join(res_dir, fn)
        if os.path.exists(p):
            print(f"  {fn}")


# =========================================================================
# メイン
# =========================================================================
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='提案パターン vs 臨床的候補パターンの複数シード比較')
    ap.add_argument('--total', type=int, default=DEFAULT_TOTAL_ELECTRODES)
    ap.add_argument('--fixed', type=int, default=DEFAULT_FIXED_ELECTRODES)
    ap.add_argument('--epoch', type=int, default=DEFAULT_EPOCH)
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                    help=f'学習シード。既定={DEFAULT_SEEDS} (5本)')
    ap.add_argument('--procs-per-gpu', type=int, default=PROCESSES_PER_GPU,
                    help='1GPUあたり並列プロセス数。全探索と併走させるなら下げる。')
    ap.add_argument('--only', nargs='+', default=None,
                    help='指定した候補名だけ実行する (スモークテスト用)')
    ap.add_argument('--analyze-only', action='store_true',
                    help='学習せず既存 CSV の集計だけ行う')
    ap.add_argument('--serial', action='store_true',
                    help='multiprocessing を使わず逐次実行する (デバッグ用)。'
                         'プール起動やGPU競合で落ちる場合の切り分けに使う。')
    ap.add_argument('--quiet-train', action='store_true',
                    help='エポックごとの学習ログを抑制する。既定では run ごとに'
                         '  [候補/seedNN] Epoch 001/100 | Train Acc: .. | Val Acc: .. | Usage: ..ch'
                         'を出す (進捗が見えないと生死判定ができないため)。')
    args = ap.parse_args()
    verbose_train = not args.quiet_train

    TOTAL, FIXED, EPOCH = args.total, args.fixed, args.epoch
    if not (0 <= FIXED <= TOTAL <= NUM_CHANNELS):
        raise ValueError("0 <= --fixed <= --total <= 16 を満たしてください")

    res_dir = os.path.join(OUTPUT_ROOT_DIR,
                           f'candidate_results_total{TOTAL}_fix{FIXED}_eeggate')
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, 'candidates.csv')

    print(f"Total electrodes: {TOTAL} / Fixed: {FIXED} / Epoch: {EPOCH}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {res_dir}\n")
    print("Candidates:")
    cands = resolve_candidates(FIXED)
    if args.only:
        cands = {k: v for k, v in cands.items() if k in args.only}
    for name, spec in cands.items():
        print(f"  {name:<20} [{spec['group']:<9}] {combo_key(spec['indices']):>14} "
              f"{'; '.join(spec['channels'])}")
    print()

    if not args.analyze_only:
        mp.set_start_method('spawn', force=True)
        _set_seed(SEED)

        # --- 中断再開: (candidate, seed) 単位で OK 済みをスキップ ---
        done = set()
        csv_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        if csv_exists:
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('status') == 'OK':
                        done.add((row['candidate'], int(row['seed'])))
            print(f"Resume: {len(done)} (candidate, seed) run(s) already done -> skip")

        pending = [(n, s) for n in cands for s in args.seeds if (n, s) not in done]
        print(f"Pending runs: {len(pending)} "
              f"({len(cands)} candidates x {len(args.seeds)} seeds)\n")

        if pending:
            print("Loading data...")
            X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes = load_data()
            if not args.serial:
                # 共有メモリ化は子プロセスにテンソルを渡すためのもの。
                # --serial では子プロセスを作らないので不要 (SIGBUS 要因を1つ減らせる)。
                for t in (X_train, y_train, X_val, y_val, X_eval, y_eval):
                    t.share_memory_()
            data_tuple = (X_train, y_train, X_val, y_val, X_eval, y_eval)
            print(f"Number of classes: {num_classes}")

            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                for g in range(num_gpus):
                    free_b, total_b = torch.cuda.mem_get_info(g)
                    print(f"  GPU{g}: free {free_b/2**30:.1f} / {total_b/2**30:.1f} GiB")
            num_workers = 1 if args.serial else (
                min(len(pending), max(1, num_gpus * max(1, args.procs_per_gpu)))
                if num_gpus > 0 else 1)
            print(f"Detected {num_gpus} GPU(s) -> {num_workers} "
                  f"{'process (SERIAL mode)' if args.serial else 'parallel worker(s)'}")

            tasks = [(i, n, cands[n], s) for i, (n, s) in enumerate(pending)]

            fcsv = open(csv_path, 'a', newline='')
            writer = csv.DictWriter(fcsv, fieldnames=FIELDNAMES, extrasaction='ignore')
            if not csv_exists:
                writer.writeheader()
                fcsv.flush()

            def _emit(r, processed, t0):
                writer.writerow(r)
                fcsv.flush()
                el = (time.perf_counter() - t0) / 60
                eta = el / processed * (len(tasks) - processed)
                print(f"[{processed}/{len(tasks)}] {r['candidate']}/seed{r['seed']} | "
                      f"BACC={r['eval_bacc']} F1={r['eval_f1']} "
                      f"Kappa={r['eval_kappa']} | {r['status']} | "
                      f"elapsed={el:.1f}min ETA={eta:.1f}min", flush=True)

            t0 = time.perf_counter()
            processed = 0
            try:
                if args.serial:
                    # デバッグ用: multiprocessing / spawn / 共有メモリを一切使わず
                    # 同一プロセスで逐次実行する。ここで落ちるならプール層は無罪。
                    _init_worker(data_tuple, num_classes, TOTAL, FIXED, EPOCH,
                                 num_gpus, verbose_train)
                    for task in tasks:
                        r = run_one(task)
                        processed += 1
                        _emit(r, processed, t0)
                else:
                    with mp.Pool(processes=num_workers, initializer=_init_worker,
                                 initargs=(data_tuple, num_classes, TOTAL, FIXED,
                                           EPOCH, num_gpus,
                                           verbose_train)) as pool:
                        for r in pool.imap_unordered(run_one, tasks, chunksize=1):
                            processed += 1
                            _emit(r, processed, t0)
            finally:
                fcsv.close()
            print(f"\nDone: {processed} run(s). CSV: {csv_path}")

    analyze(csv_path, res_dir, FIXED)
    print("\nCandidate benchmark finished!")
