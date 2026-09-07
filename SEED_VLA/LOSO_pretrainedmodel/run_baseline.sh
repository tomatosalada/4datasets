#!/bin/bash
# ベースラインモデル（CustomEEGNet）を被験者ごとに並列実行し、GPUに割り当てるスクリプト

set -u

# --- 環境設定 ---
ROOT="/mnt/data/toshiki.ohno/EEG_fatigue/EEG_VLA"
PY="${PY:-/home/islabshi/anaconda3/envs/EEG/bin/python}"
# 先ほど作成したベースライン用のPythonスクリプトの名前を指定してください
SC="$ROOT/LOSO_pretrainedmodel/LOSO_VLA.py"
OUT="${OUT:-$ROOT/LOSO_pretrainedmodel/LOSO_Baseline_best}"

# --- 並列・GPU設定 ---
JOBS="${JOBS:-8}"     # 同時に走らせるプロセス数 (例: 8プロセス)
NGPU="${NGPU:-4}"     # 使用するGPUの枚数 (例: 0, 1, 2, 3 の4枚)
EPOCHS="${EPOCHS:-100}"

mkdir -p "$OUT/logs"

# ターゲット一覧を取得 (JSONから読み込み)
TARGETS=$($PY -c "import json;print(' '.join(map(str,json.load(open('$ROOT/processdData/paper_2class/paper_manifest.json'))['kept_subjects'])))")

echo "=================================================="
echo " ベースライン並列実行を開始します"
echo "=================================================="
echo "保存先  : $OUT"
echo "並列数  : $JOBS (GPU: $NGPU 枚に分散)"
echo "対象    : $TARGETS"
echo

START=$(date +%s)
i=0

# 各被験者についてループ
for t in $TARGETS; do
  # 同時実行数が JOBS に達していたら、空きができるまで待機
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done

  # 剰余演算 (i % NGPU) でGPUを 0 -> 1 -> 2 -> 3 -> 0... と順番に割り当て
  GPU_ID=$((i % NGPU))
  echo "[$(date +%H:%M:%S)] Target $t を開始 (GPU: $GPU_ID)"

  # CUDA_VISIBLE_DEVICES を指定してバックグラウンド実行 (&)
  CUDA_VISIBLE_DEVICES=$GPU_ID OMP_NUM_THREADS=2 \
    "$PY" "$SC" --epochs "$EPOCHS" --only-target "$t" \
      --save-path "$OUT" > "$OUT/logs/t$t.log" 2>&1 &
      
  i=$((i + 1))
done

# 全ての並列ジョブが完了するのを待機
wait

# 全foldが完了したら、結果を1つのファイルに集約
echo "[$(date +%H:%M:%S)] 全foldの学習が完了。結果を集約します..."
"$PY" "$SC" --aggregate-only --save-path "$OUT" > "$OUT/logs/aggregate.log" 2>&1

END=$(date +%s)
echo "[$(date +%H:%M:%S)] 完了！ 総所要時間: $(( (END - START) / 60 )) 分"
echo "結果レポートは $OUT/loso_report.txt を確認してください。"