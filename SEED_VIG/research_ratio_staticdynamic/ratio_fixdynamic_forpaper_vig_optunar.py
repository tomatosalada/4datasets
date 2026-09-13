"""SEED-VIG における固定電極 / 動的電極の割合を LOSO で比較する。

`research_number_of_electrode/loso_researchbestnum_vig_optunar.py` と同じ
Optuna 教師モデル、fold 別モデル構造、保存済み教師標準化統計、学生モデルの
global normalization、fold 別の学習率・batch size・weight decayを使用する。
固定/動的電極の切り分けだけを追加し、既定のfoldランキングではターゲット
セッションを学習・順位決定に使用しない。

総電極数 TOTAL のうち F 本を固定し、残り TOTAL-F 本を Gate がサンプルごとに
選択する。F=0 は全て動的、F=TOTAL は静的電極選択のベースラインである。

例:
    python research_ratio_fixdynamic/ratio_fixdynamic_forpaper_vig_optunar.py
    python research_ratio_fixdynamic/ratio_fixdynamic_forpaper_vig_optunar.py --total 5
    python research_ratio_fixdynamic/ratio_fixdynamic_forpaper_vig_optunar.py \
        --only-fix 2 --only-target 9_20151017_night --epochs 1
"""

import argparse
import csv
import glob
import json
import os
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
from scipy.interpolate import griddata
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

from select_channel.select_net_class_optunar import gcn_select_net
from model.EEGNet import CustomEEGNet

warnings.filterwarnings("ignore")

# =========================================================================
# 設定
# =========================================================================
DATA_DIR = '/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/processedData/subject_wise_2class'

# LOSO fold ごとの Optuna 教師モデル、設定JSON、標準化統計量の置き場。
# CHEAT_SHEET_DIR で別の LOSO_VIG_optunar.py 出力先に変更できる。
CHEAT_SHEET_DIR = os.environ.get(
    'CHEAT_SHEET_DIR',
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'optunar_LOSO',
        'LOSO_GCN_Optuna_InnerHoldout3_SEEDVIG')))

CHEAT_SHEET_TAG = '_'.join(
    [p for p in CHEAT_SHEET_DIR.rstrip('/').split('/')[-2:] if p])

RESEARCH_BASE_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/'
                     'research_ratio_fixdynamic/'
                     'results_ratio_seedvig_2class_optuna_2/')

NUM_CLASSES = 2
CLASS_NAMES = ['Awake', 'Fatigue']
CLASS_LABELS = {0: 'Awake', 1: 'Fatigue'}

# SEED-VIG の 'chn' の並びから CPZ を除いた17ch。
# 前処理 (DE_3D_Feature_username.py) およびグラフ (GNN_graph_allset.py) と同一順序。
CHANNEL_NAMES = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8',
                 'CP1', 'CP2', 'P1', 'PZ', 'P2',
                 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']
NUM_CHANNELS = len(CHANNEL_NAMES)

# DE特徴の周波数帯数 (delta/theta/alpha/beta/gamma)。入力 [N, 16, 5, 17] の 3次元目。
NUM_BANDS = 5

# LOSO の対象となる全セッション。ファイル名 eeg_<ID>.npy / label_<ID>.npy の <ID> と一致。
SUBJECT_LIST = [
    '1_20151124_noon_2', '2_20151106_noon', '3_20151024_noon', '4_20151105_noon',
    '4_20151107_noon', '5_20141108_noon', '5_20151012_night', '6_20151121_noon',
    '7_20151015_night', '8_20151022_noon', '9_20151017_night', '10_20151125_noon',
    '11_20151024_night', '12_20150928_noon', '13_20150929_noon', '14_20151014_night',
    '15_20151126_night', '16_20151128_night', '17_20150925_noon', '18_20150926_noon',
    '19_20151114_noon', '20_20151129_night', '21_20151016_noon'
]
# 乱数種を被験者IDから決めるための通し番号。文字列hashは PYTHONHASHSEED でプロセス毎に
# 変わり再現性が壊れるため、リスト上の位置を使う。
SUBJECT_INDEX = {s: i for i, s in enumerate(SUBJECT_LIST)}

# ラベル分布から「両クラスを十分持つ被験者」を判定する閾値（少数クラスの割合）。
MIN_MINORITY_RATIO = 0.02

# Optuna教師による電極数スイープで Accuracy / Balanced Accuracy が最大の電極数。
DEFAULT_TOTAL_ELECTRODES = 6

# --ranking global 用の共通ランキング (重要度が高い順)。
# Optuna教師の loso_report.txt の colmean / clsrow を埋め込んだもの。
# ターゲット情報を含むため比較用にのみ残し、既定では fold ranking を使用する。
GLOBAL_IMPORTANCE_RANKINGS = {
    'col': ['CP1', 'P2', 'CP2', 'T7', 'PO4', 'P1', 'PO3', 'PZ', 'FT7',
            'FT8', 'T8', 'OZ', 'TP8', 'O1', 'POZ', 'O2', 'TP7'],
    'cls': ['T7', 'P2', 'CP1', 'PO3', 'T8', 'PO4', 'FT7', 'P1', 'CP2',
            'O1', 'FT8', 'O2', 'TP8', 'POZ', 'PZ', 'OZ', 'TP7'],
}
for _ranking in GLOBAL_IMPORTANCE_RANKINGS.values():
    assert sorted(_ranking) == sorted(CHANNEL_NAMES), \
        "GLOBAL_IMPORTANCE_RANKINGS が CHANNEL_NAMES と一致しません"

SEED = 42
BATCH_SIZE = 128

# 学習設定。loso_researchbestnum_vig_optunar.py と同一にする。
# F=0 (全て動的) の条件は電極数スイープの TOTAL ch と同条件でなければ比較できないので、
# learning rate / batch size / weight decay はいずれもfold別Optuna最良値を使用する。
LAMBDA_SPARSITY = 1e-3   # --select threshold のときだけ効く (topk では罰則なし)

# 全fold・全固定数で共通の学習エポック数
FIXED_EPOCHS = 100

# --- 入力の前処理とクラス重み ---
NORMALIZE = 'global'
CLASS_WEIGHT = 'none'

# --- Gate の設計 (v2_eeggate と同一。旧挙動は下の「旧」を指定すれば再現できる) ---
# 旧実装では分類器の出力がゲートのマスクだけの関数になっており、そのマスクは
# per-sample の teacher_importance が決めていて EEG はほぼ寄与していなかった。
GATE_FEAT = 'band'          # Gate に渡す EEG 特徴。band: 帯域を残して [B,5,17]->85次元
                            # mean(旧): 時間も帯域も平均して [B,17]。眠気で効く帯域差が消える
GATE_TEACHER = 'fold_const'  # 教師の電極重要度の入れ方。
                            # fold_const: 学習被験者平均の1本をfold内の全サンプルに配る
                            #             -> per-sample の通信路が消え、動くのは EEG だけ
                            # per_sample(旧): サンプルごとの重要度をそのまま渡す
                            # none: 教師を Gate 入力から外す
GATE_HINTS = False          # 教師の予測確率を Gate に入れるか。True(旧)は答えを直接渡すことになる
SELECT = 'topk'             # topk: 可変チャネルからスコア上位 (total-fixed) 本を厳密に選ぶ
                            # threshold(旧): sigmoid>0.5 + 疎性罰則。実測で目標本数に届かない
MASK_RANDOMIZE = 0.3        # 学習時にこの確率で可変チャネルのマスクをランダムなk本に
                            # 差し替える。マスクと正解の相関を壊し、分類器がマスクを
                            # 符号として読むのを防ぐ。0で無効

# 1つのGPUに載せる並列プロセス数
PROCESSES_PER_GPU = 5

# Optuna教師の fold ごとのネットワーク構造を復元するためのキー。
OPTUNA_MODEL_PARAM_KEYS = (
    'hidden_size', 'num_hidden_layers', 'transformer_dropout', 'cnn_dropout',
    'gnn_dropout', 'num_attention_heads', 'gnn_heads', 'cnn_out_channels',
)

# 電極数スイープと同一のOptuna教師キャッシュを共有する。
TEACHER_CACHE_DIR = ('/mnt/data/toshiki.ohno/EEG_fatigue/EEG_analysis_SEED-VIG/'
                     'research_number_of_electrode/'
                     'results_bestnum_seedvig_2class_optuna_2/teacher_cache')

# トポマップで補間結果を描く範囲。最寄りの実電極からこの距離を超える領域は
# 塗らずに残す（頭部円の半径1に対する比）。
#
# SEED-VIG の17chは側頭〜後頭に偏っており、前頭〜中心部には電極が1本も無い。
# 円全体を補間すると、そこに最寄り電極(FT7/FT8)の値が外挿されて
# 実測に対応しない大きな塗り領域ができてしまう。電極が無い場所は空白にする。
MAX_INTERP_DIST = 0.40

# 報告する評価指標 (参照コードと同一)
REPORT_METRICS = [('acc', 'Accuracy '),
                  ('balanced_acc', 'Balanced Accuracy'),
                  ('macro_precision', 'Precision'),
                  ('macro_recall', 'Recall   '),
                  ('macro_f1', 'F1-Score ')]
PCT_METRICS = {'acc', 'balanced_acc'}

# 頭部トポマップ上のおおよその位置（半径1の円内、+y が前方）。
# SEED-VIG の montage は側頭〜後頭に偏っているため、描画も下半分に集中する。
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
# 0. データ
# =========================================================================
def build_manifest():
    """ラベル分布から EEG_VLA 版の paper_manifest.json 相当の情報を組み立てる。

    SEED-VIG には manifest ファイルが無いので実行時に作る。判定基準は EEG_VLA 版と同じで、
    少数クラスが MIN_MINORITY_RATIO 未満のセッションは「退化」扱いにし、
    ターゲットには含めるが主指標の集計からは分けて報告する（学習には使う）。
    """
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


def load_optuna_fold_config(target):
    """LOSO_VIG_optunar.py が保存した target 固有の最良設定を読む。"""
    candidates = [
        os.path.join(CHEAT_SHEET_DIR, f'fold_target_{target}.json'),
        os.path.join(CHEAT_SHEET_DIR, f'inner_target_{target}.json'),
    ]
    config_path = next((p for p in candidates if os.path.exists(p)), None)
    if config_path is None:
        raise FileNotFoundError(
            f'fold {target} のOptuna設定が {CHEAT_SHEET_DIR} にありません。\n'
            f'必要なファイル: fold_target_{target}.json '
            f'(または inner_target_{target}.json)')

    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    params = config.get('params')
    if not isinstance(params, dict):
        raise ValueError(f'{config_path}: params がありません')
    required_params = (*OPTUNA_MODEL_PARAM_KEYS, 'learning_rate', 'batch_size',
                       'weight_decay')
    missing = [key for key in required_params if key not in params]
    if missing:
        raise ValueError(f'{config_path}: Optunaパラメータが不足しています: {missing}')
    if config.get('target') not in (None, target):
        raise ValueError(
            f'{config_path}: target={config.get("target")} は要求された {target} と不一致です')
    return config, config_path


def load_optuna_standardizer(target):
    """教師の inner train から計算・保存された mean/std を読む。"""
    path = os.path.join(CHEAT_SHEET_DIR, f'standardizer_target_{target}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'fold {target} の標準化統計量がありません: {path}\n'
            'LOSO_VIG_optunar.py の出力一式を指定してください。')
    with np.load(path) as values:
        if 'mean' not in values or 'std' not in values:
            raise ValueError(f'{path}: mean/std がありません')
        mean = torch.from_numpy(values['mean'].astype(np.float32, copy=False))
        std = torch.from_numpy(values['std'].astype(np.float32, copy=False))
    return mean, std, path


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
    """CrossEntropyLoss 用のクラス重みを作る。【学習被験者のラベルだけ】から計算する。

    loso_paper_2class_validation.py と同一実装。'balanced' は sklearn の
    class_weight='balanced' と同じ w_c = N / (C * n_c)。sum(n_c * w_c) = N なので
    平均重みが1に保たれ、損失のスケールが重みなしのときと揃う。
    """
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
    """fold 内で共通に使う教師の電極重要度 [17] を、【学習被験者だけ】から作る。

    per-sample の重要度をそのまま Gate に渡すと、それがサンプルごとの通信路になって
    マスクの中身を決めてしまう。fold 単位の定数にすれば「どの電極が有用か」という
    事前分布だけが残り、サンプルごとに動くのは EEG だけになる。
    fold_ranking() が固定電極を選ぶのに使うベクトルと同一の量。
    """
    return np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects],
                          axis=0).mean(axis=0)


def make_loader(subject_ids, shuffle, teacher_cache, normalize,
                imp_override=None, generator=None, train_mean=None, train_std=None,
                batch_size=BATCH_SIZE):
    """入力・ラベルに加えて、キャッシュ済みの教師出力も一緒に流すローダ。

    teacher_cache は build_teacher_cache() が作った npz (被験者ごとに
    hints_{s} [n,2] / imp_{s} [n,17])。load_subjects と同じ被験者順で
    連結するのでサンプルの対応はずれない。

    imp_override [17] を渡すと、教師の電極重要度を全サンプルその値に置き換える
    (GATE_TEACHER='fold_const')。キャッシュは per-sample のまま作ってあるので、
    ここで潰すだけでよく、キャッシュの作り直しは要らない。
    """
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
    assert len(hints) == len(x) and len(imp) == len(x), '教師キャッシュとサンプル数が不一致'
    return DataLoader(TensorDataset(x, y, hints, imp),
                      batch_size=batch_size, shuffle=shuffle, generator=generator)


def find_cheat_sheet(target):
    """fold (ターゲット被験者) に対応する教師モデルのパスを返す。

    そのfoldの学習被験者だけで学習された重みを使うので、ターゲット被験者は
    教師モデルにも一切入っていない。
    """
    candidates = [os.path.join(CHEAT_SHEET_DIR, f'model_target_{target}.pth'),
                  os.path.join(CHEAT_SHEET_DIR, f'best_model_target_{target}.pth')]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# =========================================================================
# 1. Gate Mechanism
# =========================================================================
class GateMechanism(nn.Module):
    """受信した EEG (+ 教師の事前分布) から、どの可変電極を使うかのスコアを出す全結合層。

    入力の内訳は呼び出し側が決める:
      feat        [B, feat_dim]  EEG 特徴。GATE_FEAT='band' なら帯域を残した 5x17=85次元
      hints       [B, 2]         教師の予測確率。GATE_HINTS=False なら渡さない
      teacher_attn[B, 17]        教師の電極重要度。GATE_TEACHER='fold_const' なら
                                 fold内で定数なので per-sample の情報は持たない
    """

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
def cls_row_importance(attention_weights):
    """CLS行 attention から電極重要度を取る (参照コードと同一)。

    attention_weights: list[layer] の list[head] の [B, 18, 18]  (CLS + 17電極)
    returns: [B, 17]  (CLSトークンを除いた17電極)
    """
    rows = []
    for layer in attention_weights:
        a = torch.stack(layer)                # [H, B, 18, 18]
        rows.append(a.mean(dim=0)[:, 0, 1:])  # ヘッド平均 -> CLS行 -> CLS列を除去
    return torch.stack(rows).mean(dim=0)      # 層平均 -> [B, 17]


@torch.no_grad()
def teacher_outputs(cheat_sheet_model, x, mode):
    """教師モデルから Gate に渡す2つの量を計算する。

    returns: teacher_hints [B, 2] (softmax確率), teacher_importance [B, 17] (電極方向のz-score)
    """
    teacher_logits, attn_w, electrode_attention, _ = cheat_sheet_model(x)
    teacher_hints = torch.softmax(teacher_logits, dim=1)

    if mode == 'cls':
        # CLS行 attention (比較用のオプション)
        teacher_imp = cls_row_importance(attn_w)                 # [B, 17]
    else:
        # 既定: 列平均。electrode_attention [Num_Layers, Batch, 18]
        #       -> 層平均 -> [CLS]トークン除外 -> [B, 17]
        teacher_imp = electrode_attention.mean(dim=0)[:, 1:]

    mean_imp = teacher_imp.mean(dim=1, keepdim=True)
    std_imp = teacher_imp.std(dim=1, keepdim=True)
    teacher_importance = (teacher_imp - mean_imp) / (std_imp + 1e-6)
    return teacher_hints, teacher_importance


def teacher_cache_path(cache_dir, mode, target, normalize):
    return os.path.join(
        cache_dir, f'teacher_{mode}_{prep_tag(normalize)}_target_{target}.npz')


@torch.no_grad()
def build_teacher_cache(target, subjects, mode, device, cache_dir, normalize):
    """Optuna教師の出力を全被験者について一度だけ計算して保存する。"""
    path = teacher_cache_path(cache_dir, mode, target, normalize)
    if os.path.exists(path):
        return path

    cheat_path = find_cheat_sheet(target)
    if cheat_path is None:
        raise FileNotFoundError(
            f"fold {target} の教師モデルが {CHEAT_SHEET_DIR} にありません。\n"
            f"先に optunar_LOSO/LOSO_VIG_optunar.py を実行してください。")

    optuna_config, config_path = load_optuna_fold_config(target)
    optuna_params = optuna_config['params']
    model_kwargs = {key: optuna_params[key] for key in OPTUNA_MODEL_PARAM_KEYS}
    model = gcn_select_net(num_classes=NUM_CLASSES, **model_kwargs).to(device)
    model.load_state_dict(torch.load(cheat_path, map_location=device))
    model.eval()

    # 教師は inner-train の保存済み統計量で標準化する。
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
    """受信した EEG から可変電極をその都度選び、固定電極と合わせて EEGNet に分類させる。

    経路は「EEG (+ fold定数の教師事前分布) -> 全結合層 -> 上位k本を選択 -> EEGNet」。

    fixed_indices  : 常時ON (ゲート重み 1.0 固定) の電極
    variable_indices: Gate がサンプルごとに 0/1 を決める電極
    n_select_var   : そのうち何本を立てるか (= total - num_fixed)

    【マスクを符号として使わせないための仕掛け】
    EEGNet の conv2 は kernel=(85,1)、つまり 電極x帯域 の85次元を一発で線形結合する。
    ゼロ埋めマスクはこの線形和に「どの行が0か」を直接見せるので、x が定数でも出力を
    マスクから復号できてしまう (旧実装ではこれが起きていた)。そこで学習時に確率
    mask_randomize で可変チャネルのマスクをランダムなk本に差し替え、マスクと正解の
    相関を壊す。差し替えた分は Gate に勾配が流れない (定数扱い) ので、分類器だけが
    「マスクの模様は当てにならない」と学ぶ。

    top-k はこの仕掛けの前提でもある。選択本数が揺れると、分類器が「立っている本数」
    だけで学習マスクとランダムマスクを見分けられてしまい、正則化が効かなくなる。
    """

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
        # 選ぶ可変チャネル数。None なら top-k を使わず閾値方式にフォールバックする
        self.n_select_var = n_select_var

        feat_dim = num_channels * n_bands if gate_feat == 'band' else num_channels
        self.gate = GateMechanism(feat_dim=feat_dim,
                                  num_variable_channels=len(variable_indices),
                                  hint_dim=num_classes if use_hints else 0,
                                  imp_dim=imp_dim)
        self.classifier = classifier_model

    def gate_features(self, x):
        """Gate に渡す EEG 特徴。x: [B, 16, 5, 17]"""
        if self.gate_feat == 'band':
            # 時間だけ平均し、帯域は残す -> [B, 5, 17] -> [B, 85]
            return x.mean(dim=1).reshape(x.size(0), -1)
        # 旧挙動: 時間も帯域も平均 -> [B, 17]
        return x.mean(dim=(1, 2))

    def _select(self, scores):
        """スコアから二値マスクを作る。forward の値は厳密に二値で、勾配は
        straight-through でスコアに流す。"""
        k = self.n_select_var
        n_var = scores.size(1)
        if self.select == 'topk' and k is not None:
            if k <= 0:
                # F=TOTAL を --full-fix gate で扱う場合。可変チャネルは1本も立てない
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
        """各サンプルについて、学習マスクと同じ本数のランダムな部分集合を作る。"""
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

        # マスクの結合処理 (Fixed=1.0 / Variable=Gate / どちらでもない電極=0.0)
        # ※ zeros 初期化 + 固定電極に1.0 を入れる形にしているので、可変チャネルが
        #    0本のとき (--full-fix static で F=TOTAL のとき) も正しく動く。
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

            # 分類器に渡すマスク。学習時のみ一部をランダムなk本に差し替える。
            # 集計・可視化に返すのは差し替え前の「学習したマスク」のほう。
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
            # 動的選定なし (固定電極だけを使う静的ベースライン)
            gate_weights_var = torch.zeros(batch_size, 0, device=device)

        x_masked = x * applied.unsqueeze(1).unsqueeze(1)
        outputs = self.classifier(x_masked)

        return outputs, gate_weights_var, full_gate_weights, teacher_importance


# =========================================================================
# 3. 固定電極の選び方 (重要度ランキング)
# =========================================================================
def fold_ranking(teacher_cache, train_subjects):
    """そのfoldの学習被験者のサンプルだけから電極重要度ランキングを作る。

    teacher_cache の imp_{s} は、そのfoldの教師 (= 学習被験者だけで学習) が出した
    サンプルごとの電極重要度 (電極方向のz-score)。学習被験者分だけを平均するので、
    ターゲット被験者の情報はランキングに一切入らない。

    returns: (ランキング(重要度の高い順の電極名), 平均重要度ベクトル [17])
    """
    imp = np.concatenate([teacher_cache[f'imp_{s}'] for s in train_subjects],
                         axis=0).mean(axis=0)                      # [17]
    order = np.argsort(imp)[::-1]
    return [CHANNEL_NAMES[i] for i in order], imp


def resolve_ranking(mode, teacher_importance, teacher_cache, train_subjects):
    """--ranking の指定に応じて、固定電極を選ぶ順序を返す。"""
    if mode == 'global':
        return list(GLOBAL_IMPORTANCE_RANKINGS[teacher_importance]), None
    return fold_ranking(teacher_cache, train_subjects)


# =========================================================================
# 4. 可視化
# =========================================================================
def plot_head_map(ax, channel_names, usage_weights, title=""):
    """電極ごとの選定確率 (0〜1) を頭部マップに補間して描く。"""
    x_coords, y_coords, z_values = [], [], []
    for i, name in enumerate(channel_names):
        if name in COORDS:
            x, y = COORDS[name]
            x_coords.append(x)
            y_coords.append(y)
            z_values.append(usage_weights[i])

    elec_x = np.array(x_coords)
    elec_y = np.array(y_coords)
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    z_values = np.array(z_values)

    # 外周は最も近い電極の値で埋める (端の外挿が破綻しないように)
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

    # 頭部円の外側は描かない
    dist = np.sqrt(grid_x ** 2 + grid_y ** 2)
    # 実電極から離れすぎた領域も描かない。SEED-VIG は前頭〜中心部に電極が無く、
    # そこを塗ると実測に対応しない外挿値を表示してしまうため。
    nearest = np.min(np.sqrt((grid_x[..., None] - elec_x) ** 2
                             + (grid_y[..., None] - elec_y) ** 2), axis=-1)
    grid_z = np.ma.masked_where((dist > 1.0) | (nearest > MAX_INTERP_DIST), grid_z)

    levels = np.linspace(0.0, 1.0, 100)
    im = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='jet', vmin=0.0, vmax=1.0)

    ax.add_artist(plt.Circle((0, 0), 1.0, color='k', fill=False, linewidth=2))
    ax.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k', linewidth=2)

    # 電極が密集しているので、マーカーの上にラベルが隠れないよう zorder を分ける
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
    """電極ごとの選定率。固定電極として使われた割合 (fold間) も重ねて示す。"""
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
    """ターゲット被験者での評価。ゲートの選定状況もクラス別に集計する。"""
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
        'balanced_acc': 100.0 * balanced_accuracy_score(trues, preds),
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
             full_fix_mode, device, fix_dir, cache_dir,
             normalize=NORMALIZE, class_weight=CLASS_WEIGHT,
             gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER, gate_hints=GATE_HINTS,
             select=SELECT, mask_randomize=MASK_RANDOMIZE):
    """総電極数 total・固定数 num_fixed・ターゲット被験者 target の LOSO fold を1本実行する。"""
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

    # 固定数によらず同一シードにして、固定数間の比較を公平にする
    seed = SEED + SUBJECT_INDEX[target]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 学生モデルのglobal normalizationはターゲットを除くouter-trainだけで計算する。
    train_mean, train_std = None, None
    if normalize == 'global':
        x_train_raw, _ = load_subjects(train_subjects, 'none')
        train_mean = x_train_raw.mean(dim=0, keepdim=True)
        train_std = x_train_raw.std(dim=0, keepdim=True)
        del x_train_raw

    tag = f'[T{total}/F{num_fixed}/{target}]'

    # --- 教師出力: このfoldの学習被験者だけで学習された教師のキャッシュ ---
    # (キャッシュが無ければここで作る。--only-target 単体実行でも動くように)
    cache_path = build_teacher_cache(target, kept, teacher_importance, device, cache_dir,
                                     normalize)
    teacher_cache = np.load(cache_path)

    # 教師の電極重要度: fold_const なら【学習被験者だけ】の平均を全サンプルに配る
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
    #   重要度ランキング上位 num_fixed 本を固定電極 (常時ON) にし、
    #   残りの電極から Gate が (total - num_fixed) 本を動的に選ぶ。
    ranking, ranking_scores = resolve_ranking(
        ranking_mode, teacher_importance, teacher_cache, train_subjects)
    fixed_names = ranking[:num_fixed]
    fixed_indices = [CHANNEL_NAMES.index(n) for n in fixed_names]
    target_var_electrodes = max(0, total - num_fixed)

    if num_fixed >= total and full_fix_mode == 'static':
        # F = TOTAL: 動的選定を完全に無くし、上位 TOTAL 本だけを使う静的ベースライン。
        # (gate を残して目標0本にするやり方だと、スパース化が弱いぶん
        #  実際には TOTAL 本より多く選ばれてしまい、比較の端点として使えない)
        variable_indices = []
    else:
        variable_indices = [i for i in range(NUM_CHANNELS) if i not in fixed_indices]

    print(f"{tag} train {len(train_subjects)}名 {train_subjects} / test [{target}] "
          f"| fixed={fixed_names} | var_target={target_var_electrodes}"
          f"/{len(variable_indices)} | epochs={epochs} "
          f"| normalize={normalize} | class_weight={class_weight} "
          f"| lr_gate=lr_eegnet={learning_rate:.6g} | batch={batch_size} "
          f"| wd={weight_decay:.6g} "
          f"| gate_feat={gate_feat} | gate_teacher={gate_teacher} | gate_hints={gate_hints} "
          f"| select={select} | mask_rand={mask_randomize} | {device}", flush=True)
    print(f"{tag} cheat sheet: {os.path.basename(find_cheat_sheet(target))} "
          f"(cached: {os.path.basename(cache_path)}, ranking: {ranking_mode})", flush=True)

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
    # クラス重みは【学習被験者のラベルだけ】から作る（ターゲットは見ない）
    weights = class_weights(train_subjects, class_weight, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    train_loader = make_loader(train_subjects, shuffle=True, teacher_cache=teacher_cache,
                               normalize=normalize, imp_override=imp_override,
                               generator=generator, train_mean=train_mean,
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
                # top-k は常に厳密に k 本なので疎性罰則は不要 (罰則ゼロと同じ)
                loss = loss_cls
            else:
                # スパース化は可変チャネルにだけ効かせる (固定チャネルは常時ONなので対象外)
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

    # --- ターゲット被験者で最終評価 (1回だけ) ---
    test_loader = make_loader([target], shuffle=False, teacher_cache=teacher_cache,
                              normalize=normalize, imp_override=imp_override,
                              train_mean=train_mean, train_std=train_std,
                              batch_size=batch_size)
    tm, trues, preds, gate_sum, gate_counts = evaluate(model, test_loader, device, loss_fn)

    print(f"{tag} test 覚醒{tm['n_class0_true']}/疲労{tm['n_class1_true']}"
          f"{'' if tm['both_classes_present'] else ' ※単一クラスのため指標が退化'} -> "
          f"Acc {tm['acc']:.2f}% | BalAcc {tm['balanced_acc']:.2f}% "
          f"| Prec {tm['macro_precision']:.4f} "
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
              'teacher_dir': CHEAT_SHEET_DIR, 'normalize': normalize,
              'class_weight': class_weight,
              'gate_feat': gate_feat, 'gate_teacher': gate_teacher,
              'gate_hints': gate_hints, 'select': select,
              'mask_randomize': mask_randomize,
              'lr_gate': learning_rate,
              'lr_classifier': learning_rate,
              'batch_size': batch_size,
              'weight_decay': weight_decay,
              'teacher_optuna_params': optuna_config['params'],
              'teacher_optuna_trial': optuna_config.get(
                  'optuna_trial', optuna_config.get('trial')),
              'teacher_optuna_epoch': optuna_config.get(
                  'epochs', optuna_config.get('epoch')),
              'teacher_optuna_config': optuna_config_path,
              'teacher_standardizer': teacher_standardizer_path}

    with open(os.path.join(fix_dir, f'fold_target_{target}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez(os.path.join(fix_dir, f'fold_target_{target}.npz'),
             trues=np.array(trues), preds=np.array(preds),
             gate_sum=gate_sum.numpy(), gate_counts=gate_counts.numpy(),
             fixed_indices=np.array(fixed_indices, dtype=np.int64),
             ranking_scores=(np.array(ranking_scores) if ranking_scores is not None
                             else np.zeros(NUM_CHANNELS)),
             curve_train_loss=np.array(curve_train_loss),
             curve_train_acc=np.array(curve_train_acc),
             curve_usage=np.array(curve_usage))
    torch.save(model.state_dict(),
               os.path.join(fix_dir, f'concept_dynamic_net_target_{target}.pth'))

    del model, classifier_model, optimizer
    torch.cuda.empty_cache()
    return result


# =========================================================================
# 6. ワーカー (mp.Pool から呼ばれる)
# =========================================================================
def worker(job_idx, total, num_fixed, target, kept, epochs, teacher_importance,
           ranking_mode, full_fix_mode, num_gpus, base_dir, cache_dir,
           normalize=NORMALIZE, class_weight=CLASS_WEIGHT,
           gate_feat=GATE_FEAT, gate_teacher=GATE_TEACHER, gate_hints=GATE_HINTS,
           select=SELECT, mask_randomize=MASK_RANDOMIZE):
    gpu_id = job_idx % num_gpus if num_gpus > 0 else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    fix_dir = os.path.join(base_dir, f'total{total}', f'fix{num_fixed}')
    os.makedirs(fix_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 1本失敗しても他の集計を止めない
    try:
        return run_fold(total, num_fixed, target, kept, epochs, teacher_importance,
                        ranking_mode, full_fix_mode, device, fix_dir, cache_dir,
                        normalize, class_weight,
                        gate_feat, gate_teacher, gate_hints, select, mask_randomize)
    except Exception as e:
        print(f"[T{total}/F{num_fixed}/{target}] FAILED: {e}", flush=True)
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
    """total{N}/fix{F}/ の fold 結果を集約して、その固定数の成績を返す。"""
    fix_dir = os.path.join(base_dir, f'total{total}', f'fix{num_fixed}')
    files = sorted(glob.glob(os.path.join(fix_dir, 'fold_target_*.json')))
    if not files:
        return None

    balanced = manifest['loso_target_subjects']

    fold_results, arrays = {}, {}
    for p in files:
        r = json.load(open(p))
        arr = np.load(p.replace('.json', '.npz'))
        # 旧実験結果にも保存済み予測から Balanced Accuracy を補完できるようにする。
        r['balanced_acc'] = 100.0 * balanced_accuracy_score(arr['trues'], arr['preds'])
        fold_results[r['target']] = r
        arrays[r['target']] = arr
    # 被験者IDは文字列なので辞書順ではなく SUBJECT_LIST の並びで揃える
    targets = sorted(fold_results, key=lambda s: SUBJECT_INDEX.get(s, len(SUBJECT_LIST)))
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
    }
    report = classification_report(global_true, global_pred,
                                   target_names=CLASS_NAMES, digits=4, zero_division=0)

    # --- 電極選定率 (fold横断) ---
    gate_sum = np.sum([arrays[t]['gate_sum'] for t in targets], axis=0)        # [2, 17]
    gate_counts = np.sum([arrays[t]['gate_counts'] for t in targets], axis=0)  # [2]
    denom = gate_counts.copy()
    denom[denom == 0] = 1.0
    class_distribution = gate_sum / denom[:, None]
    overall_usage = gate_sum.sum(axis=0) / max(gate_counts.sum(), 1.0)

    # --- 各電極が「固定電極」に選ばれた fold の割合 ---
    #     --ranking fold ではfoldごとに固定電極が変わるので、その安定性を見る
    fixed_rate = np.zeros(NUM_CHANNELS)
    for t in targets:
        fixed_rate[arrays[t]['fixed_indices'].astype(int)] += 1.0
    fixed_rate /= max(len(targets), 1)

    # --- 可視化 ---
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

    # --- テキスト保存 ---
    ratio = (num_fixed / total) if total > 0 else 0.0
    learning_rates = sorted({float(fold_results[t]['lr_gate']) for t in targets
                             if fold_results[t].get('lr_gate') is not None})
    batch_sizes = sorted({
        int(fold_results[t].get(
            'batch_size', fold_results[t].get('teacher_optuna_params', {}).get('batch_size')))
        for t in targets
    })
    weight_decays = [
        float(fold_results[t].get(
            'weight_decay',
            fold_results[t].get('teacher_optuna_params', {}).get('weight_decay')))
        for t in targets
    ]
    with open(os.path.join(fix_dir, 'loso_report.txt'), 'w') as f:
        f.write(f"===== LOSO 2-class (SEED-VIG subject_wise_2class) / total {total}ch, "
                f"fixed {num_fixed}ch ({ratio*100:.1f}%) =====\n")
        f.write("metrics: Accuracy / Balanced Accuracy / Precision / Recall / F1-Score "
                "(Precision, Recall, F1 は macro 平均)\n")
        f.write(f"epochs : {fold_results[targets[0]].get('epochs')} (全fold固定)\n")
        f.write(f"ranking: {fold_results[targets[0]].get('ranking_mode')} "
                f"(fold=学習被験者のみから作成 / global=全fold共通)\n")
        f.write(f"labeling: {manifest['labeling_rule']}\n")
        f.write("\n===== Hyperparameters =====\n")
        f.write("  lr_gate:           foldごとのOptuna learning_rate\n")
        f.write("  lr_classifier:     foldごとのOptuna learning_rate\n")
        f.write(f"  learning_rates:    {learning_rates}\n")
        f.write("  weight_decay:      foldごとのOptuna weight_decay\n")
        f.write(f"  weight_decay range: {min(weight_decays):.8g} - "
                f"{max(weight_decays):.8g}\n")
        f.write("  batch_size:        foldごとのOptuna batch_size\n")
        f.write(f"  batch_size values: {batch_sizes}\n")
        f.write(f"  lambda_sparsity:   {LAMBDA_SPARSITY} "
                f"(select=threshold のときだけ有効)\n")
        r0 = fold_results[targets[0]]
        f.write(f"  teacher_dir:       {r0.get('teacher_dir', CHEAT_SHEET_DIR)}\n")
        f.write(f"  normalize:         {r0.get('normalize')}\n")
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
            f.write(f"  {s}: n={r['n_test']} (awake={r['n_class0_true']}/"
                    f"fatigue={r['n_class1_true']}) acc={r['acc']:.2f}% "
                    f"bal_acc={r['balanced_acc']:.2f}% "
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
    """固定数ごとの結果を集約して CSV / 比較プロットを作る。"""
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
            # --- 全ターゲット ---
            'all_acc_mean': g(sa, 'acc'), 'all_acc_std': g(sa, 'acc', 'std'),
            'all_balanced_acc_mean': g(sa, 'balanced_acc'),
            'all_balanced_acc_std': g(sa, 'balanced_acc', 'std'),
            'all_precision_mean': g(sa, 'macro_precision'),
            'all_recall_mean': g(sa, 'macro_recall'),
            'all_f1_mean': g(sa, 'macro_f1'), 'all_f1_std': g(sa, 'macro_f1', 'std'),
            # --- 両クラスを持つ被験者のみ (論文の主指標) ---
            'bal_acc_mean': g(sb, 'acc'), 'bal_acc_std': g(sb, 'acc', 'std'),
            'bal_balanced_acc_mean': g(sb, 'balanced_acc'),
            'bal_balanced_acc_std': g(sb, 'balanced_acc', 'std'),
            'bal_precision_mean': g(sb, 'macro_precision'),
            'bal_recall_mean': g(sb, 'macro_recall'),
            'bal_f1_mean': g(sb, 'macro_f1'), 'bal_f1_std': g(sb, 'macro_f1', 'std'),
            # --- プール (全サンプルを一つにまとめた指標) ---
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
    print(f"{'Fix':>3} | {'ratio':>5} | {'sel':>5} | {'balAcc':>7} | {'balBAcc':>8} | {'balF1':>7} | "
          f"{'allAcc':>7} | {'allF1':>7} | {'poolAcc':>7} | folds")
    print("-" * 94)
    for r in rows:
        print(f"{r['n_fixed_electrodes']:>3} | {r['fixed_ratio']:>5.2f} | "
              f"{r['avg_selected_electrodes']:>5.2f} | "
              f"{r['bal_acc_mean']:>7.2f} | {r['bal_balanced_acc_mean']:>8.2f} | "
              f"{r['bal_f1_mean']:>7.4f} | "
              f"{r['all_acc_mean']:>7.2f} | {r['all_f1_mean']:>7.4f} | "
              f"{r['pooled_acc']:>7.2f} | {r['n_folds']}")

    # 論文の主指標である「両クラスを持つ被験者のみ」の macro F1 で最良の固定数を決める
    valid = [r for r in rows if not np.isnan(r['bal_f1_mean'])]
    if valid:
        best = max(valid, key=lambda x: x['bal_f1_mean'])
        print(f"\n>>> Best by balanced-target macro F1: fixed {best['n_fixed_electrodes']}ch "
              f"/ dynamic {best['n_variable_target']}ch (ratio={best['fixed_ratio']:.2f}, "
              f"F1={best['bal_f1_mean']:.4f} ± {best['bal_f1_std']:.4f}, "
              f"Acc={best['bal_acc_mean']:.2f}%, "
              f"BalAcc={best['bal_balanced_acc_mean']:.2f}%)")

    # --- 固定数 vs 指標 のプロット ---
    ns = [r['n_fixed_electrodes'] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    balanced_series = [
        ('bal_acc_mean', 'Accuracy', 'o', 100.0, '-'),
        ('bal_f1_mean', 'Macro F1', 's', 1.0, '-'),
        ('bal_precision_mean', 'Macro Precision', 'v', 1.0, '-'),
        # 2クラスでは Macro Recall と同値なので、Balanced Accuracy のみ表示する。
        ('bal_balanced_acc_mean', 'Balanced Accuracy', 'D', 100.0, '-'),
    ]
    for key, label, marker, scale, linestyle in balanced_series:
        ax.plot(ns, [r[key] / scale for r in rows], marker=marker, label=label,
                linestyle=linestyle, zorder=3 if key == 'bal_balanced_acc_mean' else 2)
    ax.set_xlabel(f'Number of FIXED electrodes (out of total {total})')
    ax.set_ylabel('Score')
    ax.set_title(f'LOSO metrics vs. fixed/dynamic ratio '
                 f'(SEED-VIG, fatigue 2-class, total {total}ch, balanced targets)')
    ax.set_xticks(ns)
    ax.grid(True); ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = os.path.join(total_dir, 'summary_metrics_vs_fixed.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    with open(os.path.join(total_dir, 'summary_fix_ratio.json'), 'w') as f:
        json.dump({'data_dir': DATA_DIR,
                   'cheat_sheet_dir': CHEAT_SHEET_DIR,
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
    ap = argparse.ArgumentParser(
        description='固定電極 / 動的電極の割合をスイープする LOSO 実験 (SEED-VIG 2class)')
    ap.add_argument('--total', type=int, default=DEFAULT_TOTAL_ELECTRODES,
                    help=f'総電極数 (1〜{NUM_CHANNELS})。既定 {DEFAULT_TOTAL_ELECTRODES} '
                         f'(research_number_of_electrode の最良値)')
    ap.add_argument('--epochs', type=int, default=FIXED_EPOCHS,
                    help=f'全fold共通の学習エポック数（既定 {FIXED_EPOCHS}）')
    ap.add_argument('--fix-counts', type=int, nargs='+', default=None,
                    help='探索する固定電極数（既定 0〜--total）')
    ap.add_argument('--only-fix', type=int, default=None,
                    help='この固定電極数だけ実行する')
    ap.add_argument('--only-target', type=str, default=None,
                    help='このセッション1件のfoldだけ実行する（並列実行用）。集約は行わない。'
                         '例: --only-target 9_20151017_night')
    ap.add_argument('--targets', choices=['all', 'balanced'], default='all',
                    help="all: 読み込めた全セッションをターゲットにする(既定) / "
                         "balanced: 少数クラスが2%%以上のセッションのみ")
    ap.add_argument('--limit-folds', type=int, default=None,
                    help='先頭N名だけ回す（動作確認用）')
    ap.add_argument('--teacher-importance', choices=['col', 'cls'], default='col',
                    help='Gateに渡す教師の電極重要度。col=electrode_attentionの列平均(既定) / '
                         'cls=CLS行attention(比較用)')
    ap.add_argument('--normalize', choices=['none', 'subject', 'channel_band', 'global'],
                    default=NORMALIZE,
                    help=f'学生モデル入力の標準化方式（既定 {NORMALIZE}）')
    ap.add_argument('--class-weight', choices=['none', 'balanced'],
                    default=CLASS_WEIGHT,
                    help=f'CrossEntropyLoss のクラス重み（既定 {CLASS_WEIGHT}）。'
                         'balanced: 学習プールのクラス頻度の逆数 N/(C*n_c)')
    ap.add_argument('--gate-feat', choices=['band', 'mean'], default=GATE_FEAT,
                    help=f'Gate に渡す EEG 特徴（既定 {GATE_FEAT}）。'
                         'band: 時間だけ平均し帯域を残す [B,5,17]->85次元 / '
                         'mean: 時間も帯域も平均 [B,17]（旧挙動。帯域差が消える）')
    ap.add_argument('--gate-teacher', choices=['fold_const', 'per_sample', 'none'],
                    default=GATE_TEACHER,
                    help=f'教師の電極重要度の入れ方（既定 {GATE_TEACHER}）。'
                         'fold_const: 学習被験者平均をfold内の全サンプルに配る / '
                         'per_sample: サンプルごと（旧挙動。マスクの中身がこれになる）/ '
                         'none: Gate 入力から外す')
    ap.add_argument('--gate-hints', action='store_true', default=GATE_HINTS,
                    help='教師の予測確率を Gate に入れる（旧挙動）。既定は入れない')
    ap.add_argument('--select', choices=['topk', 'threshold'], default=SELECT,
                    help=f'電極の選び方（既定 {SELECT}）。'
                         'topk: 可変チャネルから上位(total-fixed)本を厳密に選ぶ / '
                         'threshold: sigmoid>0.5 + 疎性罰則（旧挙動。本数が目標に届かない）')
    ap.add_argument('--mask-randomize', type=float, default=MASK_RANDOMIZE,
                    help=f'学習時にこの確率で可変チャネルのマスクをランダムなk本に'
                         f'差し替える（既定 {MASK_RANDOMIZE}）。'
                         '分類器がマスクを符号として読むのを防ぐ。0で無効（旧挙動）')
    ap.add_argument('--full-fix', choices=['static', 'gate'], default='static',
                    help='固定数が総電極数と等しいとき(F=TOTAL)の扱い。'
                         'static=Gateを使わず上位TOTAL本だけを使う静的ベースライン(既定) / '
                         'gate=Gateを残して目標0本にする')
    ap.add_argument('--ranking', choices=['fold', 'global'], default='fold',
                    help='固定電極の選定順。fold=そのfoldの学習被験者だけから作る'
                         '(リークなし、既定) / global=全fold共通ランキング')
    ap.add_argument('--jobs', type=int, default=None,
                    help='同時実行プロセス数（既定 GPU数×%d）' % PROCESSES_PER_GPU)
    ap.add_argument('--aggregate-only', action='store_true',
                    help='学習せず、保存済みのfold結果から集約だけ行う')
    ap.add_argument('--save-path', type=str, default=RESEARCH_BASE_DIR)
    ap.add_argument('--teacher-cache-dir', type=str, default=TEACHER_CACHE_DIR,
                    help='教師出力キャッシュの置き場 '
                         '(既定はOptuna電極数スイープのキャッシュを共有)')
    args = ap.parse_args()

    if not (1 <= args.total <= NUM_CHANNELS):
        ap.error(f"--total は 1〜{NUM_CHANNELS} を指定してください (指定値: {args.total})")

    base_dir = args.save_path
    os.makedirs(base_dir, exist_ok=True)

    # SEED-VIG には manifest ファイルが無いので、ラベル分布から実行時に組み立てる。
    manifest = build_manifest()
    kept = manifest['kept_subjects']              # 読み込めた全セッション
    balanced = manifest['loso_target_subjects']   # 両クラスを十分持つセッション
    if not kept:
        raise SystemExit(f"データが見つかりません: {DATA_DIR}\n"
                         f"preprocessing_VIG/label_2class_subject_wise.py を先に実行してください")
    with open(os.path.join(base_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 固定数のスイープ範囲: 0 (全て動的) 〜 総電極数 (全て固定)
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
            raise SystemExit(f"--only-target {args.only_target} は読み込めるセッションに"
                             f"含まれていません。候補: {kept}")
        targets = [args.only_target]
    else:
        targets = kept if args.targets == 'all' else balanced
        if args.limit_folds:
            targets = targets[:args.limit_folds]

    degenerate = [s for s in targets if s not in balanced]
    num_gpus = torch.cuda.device_count()

    print(f"データ    : {DATA_DIR}")
    print(f"Optuna教師: {CHEAT_SHEET_DIR} (foldごとのモデル・最良params・標準化を使用)")
    print(f"ラベル    : {manifest['labeling_rule']}")
    print(f"欠損      : {manifest['excluded_subjects']}")
    print(f"学習可    : {len(kept)}名 {kept}")
    print(f"ターゲット: {len(targets)}名 {targets}")
    if degenerate:
        print(f"  ※ うち{len(degenerate)}名 {degenerate} は少数クラスが極端に少なく、"
              f"macro Precision/Recall/F1が退化する。集計は両方を出す")
    print(f"総電極数  : {args.total}")
    print(f"固定電極数: {fix_counts} (残り {args.total}-F 本を動的選定)")
    print(f"エポック  : {args.epochs} (全fold固定 / validationもtestによる選択も無し)")
    print("最適化設定: foldごとのOptuna learning_rate / batch_size / weight_decayを使用")
    print(f"教師重要度: {args.teacher_importance}")
    print(f"標準化    : {args.normalize}")
    print(f"クラス重み: {args.class_weight}")
    print(f"Gate      : feat={args.gate_feat} / teacher={args.gate_teacher} "
          f"/ hints={args.gate_hints} / select={args.select} "
          f"/ mask_randomize={args.mask_randomize}")
    print(f"F=TOTAL時 : {args.full_fix} "
          f"({'Gateなし・上位TOTAL本のみ' if args.full_fix == 'static' else 'Gateあり・目標0本'})")
    print(f"固定順    : {args.ranking}"
          + ("  (foldの学習被験者だけから作成)" if args.ranking == 'fold'
             else f"  {GLOBAL_IMPORTANCE_RANKINGS[args.teacher_importance]}"))

    # 事前に教師モデルの有無を確認する（全fold回してから気づくのを避ける）
    # 必要なのはターゲット自身の教師だけ（build_teacher_cache が読むのは find_cheat_sheet(target)）
    missing = [t for t in targets if find_cheat_sheet(t) is None]
    if missing:
        print(f"\n【エラー】教師モデルが見つからないセッション: {missing}")
        print(f"  {CHEAT_SHEET_DIR} に model_target_<session>.pth が必要です。")
        print("  先に optunar_LOSO/LOSO_VIG_optunar.py を実行してください。")
        sys.exit(1)

    try:
        for target in targets:
            load_optuna_fold_config(target)
            load_optuna_standardizer(target)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f'\n【エラー】Optuna教師の出力一式を確認してください:\n  {exc}') from exc

    # 教師出力を先に1回だけ計算しておく。以降 (固定数 x エポック) の全学習で使い回す。
    cache_dir = args.teacher_cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n教師出力キャッシュ (共有): {cache_dir}", flush=True)
    for t in targets:
        p = teacher_cache_path(cache_dir, args.teacher_importance, t, args.normalize)
        existed = os.path.exists(p)
        build_teacher_cache(t, kept, args.teacher_importance, cache_device, cache_dir,
                            args.normalize)
        print(f"  {t}: {os.path.basename(p)}"
              f"{' (既存)' if existed else ''}", flush=True)

    # 固定電極ランキング (fold ごと) を先に表示して、どこを固定するのか確認できるようにする
    if args.ranking == 'fold':
        print("\nfoldごとの固定電極 (重要度上位):", flush=True)
        for t in targets:
            cache = np.load(teacher_cache_path(
                cache_dir, args.teacher_importance, t, args.normalize))
            rk, _ = fold_ranking(cache, [s for s in kept if s != t])
            print(f"  {t}: {rk[:args.total]}", flush=True)

    jobs = [(fx, t) for fx in fix_counts for t in targets]

    if args.jobs is not None:
        num_workers = max(1, args.jobs)
    elif num_gpus > 0:
        num_workers = max(1, num_gpus * PROCESSES_PER_GPU)
    else:
        num_workers = 1   # CPU実行時は逐次
    num_workers = min(num_workers, len(jobs))
    print(f"\n検出GPU   : {num_gpus} -> {num_workers} 並列 / 全 {len(jobs)} fold\n", flush=True)

    args_list = [(i, args.total, fx, t, kept, args.epochs, args.teacher_importance,
                  args.ranking, args.full_fix, num_gpus, base_dir, cache_dir,
                  args.normalize, args.class_weight,
                  args.gate_feat, args.gate_teacher, args.gate_hints,
                  args.select, args.mask_randomize)
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
