#!/usr/bin/env python3
"""summary_candidates.json から候補比較のバー図 (Sub10を除く14名版) を描き直す。

  * 指標は Sub10 を除外した 14名 のもの。バー = 14名の平均±SD、点 = 各 fold (14個)。
  * 候補 'proposed' を除外せずにそのまま表示する。
  * パネルは左から Accuracy(%) -> Macro F1 -> Cohen's Kappa。
  * X軸の起点はすべての指標で 0.0 に固定されます。

usage:
    python replot_candidates_metrics_14targets.py
    (引数なしの場合、指定されたパスの summary_candidates.json を読み込みます)
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# パネルの並び順
FIG_METRICS = [('acc', 'Accuracy(%)'),
               ('macro_f1', 'Macro F1'),
               ('kappa', "Cohen's Kappa")]

GROUP_COLORS = {'proposed': '#c0504d', 'literature': '#3b6fb6',
                'control': '#9aa4b1', 'random': '#7fa66b'}

# proposed を除外しないようにし、名前の置き換えも解除
DROP = set()            
RENAME = {}             


def replot_metrics(res_dir):
    json_path = os.path.join(res_dir, 'research_fix_electrode/results_candidates_static4/total10_fix4/summary_candidates.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} が見つかりません。")
        return

    with open(json_path) as f:
        js = json.load(f)
        
    details = [r for r in js['per_candidate'] if r['candidate'] not in DROP]
    total, fixed = js['n_total_electrodes'], js['n_fixed_electrodes']

    disp = {r['candidate']: RENAME.get(r['candidate'], r['candidate'])
            for r in details}
    stat_of = {disp[r['candidate']]: r for r in details}
    per_fold = {disp[r['candidate']]: r['per_subject'] for r in details}
    
    # ★修正点1: 対象被験者を Sub10 を除く 14名 に設定
    first_candidate = next(iter(per_fold.keys()))
    target_subjects = [s for s in per_fold[first_candidate].keys() if str(s) != "10"]
    n_folds = len(target_subjects)

    # 14名分の平均・標準偏差を直接計算するヘルパー関数
    def get_14sub_stats(candidate, metric_key):
        scores = [per_fold[candidate][subj][metric_key] 
                  for subj in target_subjects 
                  if metric_key in per_fold[candidate][subj] and per_fold[candidate][subj][metric_key] is not None]
        if scores:
            return np.mean(scores), np.std(scores, ddof=0), scores
        return None, None, []

    plt.rcParams.update({'figure.dpi': 130, 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, len(FIG_METRICS),
                             figsize=(4.4 * len(FIG_METRICS), 5.0))
                             
    for ax, (key, label) in zip(np.atleast_1d(axes), FIG_METRICS):
        vals = []
        for n in stat_of:
            # ★修正点2: JSONの集計値ではなく、14名分の再計算値を利用
            mean_val, std_val, pts = get_14sub_stats(n, key)
            if mean_val is not None:
                vals.append((n, mean_val, std_val, pts))
        
        # 平均が同値の場合は他指標 (Macro F1 -> Kappa) の良い方を上位に置く。
        def _tiebreak(item):
            n, mean_val, std_val, pts = item
            f1_mean, _, _ = get_14sub_stats(n, 'macro_f1')
            kappa_mean, _, _ = get_14sub_stats(n, 'kappa')
            return (round(mean_val, 6), 
                    f1_mean if f1_mean is not None else float('-inf'), 
                    kappa_mean if kappa_mean is not None else float('-inf'))
            
        vals.sort(key=_tiebreak)
        names = [item[0] for item in vals]
        means = np.array([item[1] for item in vals])
        sds = np.array([item[2] for item in vals])
        pts_list = [item[3] for item in vals]
        
        cols = [GROUP_COLORS.get(stat_of[n].get('group', 'literature'), '#888') for n in names]
        y = np.arange(len(names))
        
        ax.barh(y, means, xerr=sds, color=cols, alpha=.9,
                error_kw=dict(lw=1, capsize=2.5, ecolor='#444'))
                
        for yi, pts in zip(y, pts_list):
            if pts:
                ax.scatter(pts, np.full(len(pts), yi, dtype=float), s=8,
                           color='black', alpha=.45, zorder=3)
                           
        ax.set_yticks(y, names)
        if len(means) > 0:
            ax.set_xlim(0.0, float(means.max() + 2 * (sds.max() + 1e-3)))
            
        ax.set_xlabel(label)
        ax.set_title(label)
        
    # タイトルと出力ファイル名も14名用に変更
    fig.suptitle(f'Proposed vs. fatigue-motivated fixed electrode sets\n'
                 f'(SEED-VLA, total {total}ch / fixed {fixed}ch, '
                 f'14 targets without Sub10 (n={n_folds}); dots = individual folds)', y=1.03)
    fig.tight_layout()
    
    out = os.path.join(res_dir, 'research_fix_electrode/results_candidates_static4/total10_fix4/fig_candidates_metrics_14targets.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    target_dirs = sys.argv[1:] if len(sys.argv) > 1 else ['.']
    for d in target_dirs:
        replot_metrics(d)