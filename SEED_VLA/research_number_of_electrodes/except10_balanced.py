"""summary_all_channels_VLA.json から「Sub10を除いた14名」の電極数 vs 指標プロットを作成する。

usage: python plot_VLA_without_sub10.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    file_path = "research_number_of_electrode/results_bestnum/summary_all_channels.json"
    
    # JSONファイルの読み込み
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 抽出結果を格納する辞書
    metrics_means = {
        "n_target": [],
        "Acc": [],
        "Prec": [],
        "Rec": [],
        "F1": []
    }

    # 各チャネル設定（1〜18）ごとに計算
    for count_data in data["per_count"]:
        n_target = count_data["n_target_electrodes"]
        metrics_means["n_target"].append(n_target)
        
        acc_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        
        # Sub10 を除外した被験者リストを作成
        target_subjects = [str(s) for s in count_data["targets"] if str(s) != "10"]
        
        # 除外後の被験者ごとのスコアを取得
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

    ns = metrics_means["n_target"]
    # Accuracy は % 表記のため 100 で割ってスケールを 0.0 - 1.0 に調整
    ax.plot(ns, [a / 100.0 for a in metrics_means["Acc"]], marker='d', label='Accuracy')
    ax.plot(ns, metrics_means["F1"], marker='^', label='Macro F1')
    ax.plot(ns, metrics_means["Prec"], marker='v', label='Macro Precision')
    ax.plot(ns, metrics_means["Rec"], marker='<', label='Macro Recall')

    ax.set_xlabel('Target number of electrodes')
    ax.set_ylabel('Score')
    ax.set_title('LOSO metrics vs. number of electrodes\n(SEED-VLA, fatigue 2-class, 14 targets without Sub10)')
    ax.set_xticks(ns)
    ax.grid(True)
    ax.legend(fontsize=9)
    fig.tight_layout()

    # 画像として保存
    output_file = "research_number_of_electrode/results_bestnum_paper2class_eeggate_baseline_final_a/summary_metrics_vs_channels_VLA_without_sub10.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f'saved: {output_file} (n_subjects={len(target_subjects)})')

if __name__ == '__main__':
    main()