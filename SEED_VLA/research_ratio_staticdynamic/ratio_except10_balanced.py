"""summary_fix_ratio_VLA.json から「Sub10を除き、かつ両クラスを持つ被験者（balanced）」の固定電極数 vs 指標プロットを作成する。

usage: python plot_fixratio_VLA_balanced_without_sub10.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    file_path = "research_ratio_fixdynamic/results_ratio/total10/summary_fix_ratio.json"
    
    # JSONファイルの読み込み
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 抽出結果を格納する辞書
    metrics_means = {
        "n_fixed": [],
        "Acc": [],
        "Prec": [],
        "Rec": [],
        "F1": []
    }

    # 各固定電極数（fix_counts: 0〜12）ごとに計算
    for count_data in data["per_fix_count"]:
        n_fixed = count_data["n_fixed_electrodes"]
        metrics_means["n_fixed"].append(n_fixed)
        
        acc_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        
        # Sub10 のみを除外した被験者リストを作成（残り14名）
        target_subjects = [str(s) for s in count_data["targets"] if str(s) != "10"]
        
        # 該当被験者のスコアを抽出
        for sub in target_subjects:
            if sub in count_data["per_subject"]:
                scores = count_data["per_subject"][sub]
                acc_list.append(scores["acc"])
                prec_list.append(scores["macro_precision"])
                rec_list.append(scores["macro_recall"])
                f1_list.append(scores["macro_f1"])
                
        # 各指標の平均を計算して保存
        metrics_means["Acc"].append(np.mean(acc_list))
        metrics_means["Prec"].append(np.mean(prec_list))
        metrics_means["Rec"].append(np.mean(rec_list))
        metrics_means["F1"].append(np.mean(f1_list))

    # グラフ描画
    fig, ax = plt.subplots(figsize=(10, 6))

    ns = metrics_means["n_fixed"]
    # Accuracy は % 表記のため 100 で割ってスケールを 0.0 - 1.0 に調整
    ax.plot(ns, [a / 100.0 for a in metrics_means["Acc"]], marker='d', label='Accuracy')
    ax.plot(ns, metrics_means["F1"], marker='^', label='Macro F1')
    ax.plot(ns, metrics_means["Prec"], marker='v', label='Macro Precision')
    ax.plot(ns, metrics_means["Rec"], marker='<', label='Macro Recall')

    total_electrodes = data.get("n_total_electrodes", 12)
    ax.set_xlabel(f'Number of FIXED electrodes (out of total {total_electrodes})')
    ax.set_ylabel('Score')
    ax.set_title(f'LOSO metrics vs. fixed/dynamic ratio\n(SEED-VLA, fatigue 2-class, total {total_electrodes}ch, balanced targets without Sub10)')
    ax.set_xticks(ns)
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()

    # 画像として保存
    output_file = "research_ratio_fixdynamic/results_ratio/total10/summary_metrics_vs_fixed_VLA_balanced_without_sub10.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'saved: {output_file} (n_subjects={len(target_subjects)})')

if __name__ == '__main__':
    main()