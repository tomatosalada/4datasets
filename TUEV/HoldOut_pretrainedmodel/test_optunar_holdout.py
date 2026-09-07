import sys
import os
import torch
import torch.nn as nn
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg') # Ensure multiprocessing safety for plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, cohen_kappa_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import optuna
import copy
import multiprocessing
from select_channel.select_net_class_optunar import *

class DualLogger(object):
    def __init__(self, filename="result_optunar_holdout.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# sys.stdoutとsys.stderrを書き換えて、コンソールとファイル両方に出力する
sys.stdout = DualLogger("result_optunar_holdout.log")
sys.stderr = sys.stdout

warnings.filterwarnings("ignore")

# saveパス
save_path = "/mnt/data/toshiki.ohno/TUEV_dataset/TU1v/TU1v/v2.0.1/test_classification/result_optuna_holdout/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 固定パラメータ
epoch = 100
epochs = list(range(1, epoch + 1))
seed = 42

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data():
    data_dir = "/mnt/data/toshiki.ohno/TUEV_dataset/TU1v/TU1v/v2.0.1/TUEV/freq_formal_data/"
    
    train_dir = os.path.join(data_dir, "processed_train_split")
    val_dir = os.path.join(data_dir, "processed_val_split")
    eval_dir = os.path.join(data_dir, "processed_eval_split")

    X_train_np = np.load(os.path.join(train_dir, "train_data.npy"))
    y_train_np = np.load(os.path.join(train_dir, "train_labels.npy")).squeeze() - 1

    X_val_np = np.load(os.path.join(val_dir, "train_data.npy"))
    y_val_np = np.load(os.path.join(val_dir, "train_labels.npy")).squeeze() - 1

    X_eval_np = np.load(os.path.join(eval_dir, "eval_data.npy"))
    y_eval_np = np.load(os.path.join(eval_dir, "eval_labels.npy")).squeeze() - 1

    num_classes = len(np.unique(y_train_np))

    # モデルの入力形式 [batch, seq_len, freq_len, node_dim] に変換
    # 現在の形状 [N, 16(channel), 10(time), 5(freq)] -> [N, 10(time), 5(freq), 16(channel)]
    X_train_np = np.transpose(X_train_np, (0, 2, 3, 1))
    X_val_np = np.transpose(X_val_np, (0, 2, 3, 1))
    X_eval_np = np.transpose(X_eval_np, (0, 2, 3, 1))

    # Data Leakageを防ぐため、ここでは正規化を行わずNumpy配列のまま返す
    # 正規化は学習ループ内（Trainデータのみを基準）で実行する
    return X_train_np, y_train_np, X_val_np, y_val_np, X_eval_np, y_eval_np, num_classes

# TUEVの16チャンネル
channel_names = [
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
]

# 各電極の配置座標 (頭部のマップ描画用)
coords = {
    'FP1-F7': (-0.6, 0.7), 'F7-T3': (-0.8, 0.3), 'T3-T5': (-0.8, -0.3), 'T5-O1': (-0.6, -0.7),
    'FP2-F8': (0.6, 0.7), 'F8-T4': (0.8, 0.3), 'T4-T6': (0.8, -0.3), 'T6-O2': (0.6, -0.7),
    'FP1-F3': (-0.3, 0.7), 'F3-C3': (-0.3, 0.0), 'C3-P3': (-0.3, -0.5), 'P3-O1': (-0.3, -0.7),
    'FP2-F4': (0.3, 0.7), 'F4-C4': (0.3, 0.0), 'C4-P4': (0.3, -0.5), 'P4-O2': (0.3, -0.7)
}

# --- 可視化用の関数定義 ---
def visualize_attention(attention_matrix, acc):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=channel_names, yticklabels=channel_names, cmap="viridis", square=True)
    plt.title(f"Best Attention (Val Acc: {acc:.2f}%)")
    plt.xlabel("Destination")
    plt.ylabel("Source")
    plt.savefig(f"{save_path}attention_heatmap.png")
    plt.close()

    pos = np.array([coords.get(name, (0, 0)) for name in channel_names])
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2)
    ax.add_artist(circle)
    plt.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k-', linewidth=2)

    for i, name in enumerate(channel_names):
        x, y = pos[i]
        plt.scatter(x, y, s=400, c='white', edgecolors='black', zorder=10)
        plt.text(x, y, name, ha='center', va='center', fontsize=7, fontweight='bold', zorder=11)

    threshold = np.percentile(attention_matrix, 95)
    for i in range(len(channel_names)):
        for j in range(len(channel_names)):
            weight = attention_matrix[i, j]
            if weight > threshold and i != j:
                p1, p2 = pos[i], pos[j]
                alpha = (weight - threshold) / (attention_matrix.max() - threshold + 1e-9)
                alpha = np.clip(alpha, 0.1, 1.0)
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='red', alpha=alpha, linewidth=3*alpha, zorder=1)

    plt.title(f"Strongest Connections (Top 5%)")
    plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
    plt.axis('off')
    plt.savefig(f"{save_path}attention_headmap.png")
    plt.close()

def visualize_electrode_importance(importance_vector, acc):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=channel_names, y=importance_vector, palette="viridis")
    plt.title(f"Electrode Importance (Val Acc: {acc:.2f}%)")
    plt.xlabel("Channel")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close()

import threading

# スレッドごとのGPU割り当てを管理するグローバル変数
thread_to_gpu = {}
gpu_assign_lock = threading.Lock()
next_gpu_index = 0

# --- 学習処理 ---
def run_trial(params, X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes, trial_number=0):
    set_seed(seed)
    
    learningRate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    hidden_size = params["hidden_size"]
    num_hidden_layers = params["num_hidden_layers"]
    transformer_dropout = params["transformer_dropout"]
    cnn_dropout = params["cnn_dropout"]
    gnn_dropout = params["gnn_dropout"]
    num_attention_heads = params["num_attention_heads"]
    gnn_heads = params["gnn_heads"]
    cnn_out_channels = params.get("cnn_out_channels", 16)
    use_lr_scheduler = params.get("use_lr_scheduler", False)
    save_results = params.get("save_results", False)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            global thread_to_gpu, next_gpu_index
            thread_id = threading.get_ident()
            with gpu_assign_lock:
                if thread_id not in thread_to_gpu:
                    thread_to_gpu[thread_id] = next_gpu_index % num_gpus
                    next_gpu_index += 1
            gpu_id = thread_to_gpu[thread_id]
        else:
            gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        
    print(f"\n======================================")
    print(f"          Starting Holdout Trial on {device}")
    print(f"======================================")
    
    # --- Data Leakage防止：Trainデータのみで正規化を計算 ---
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_val_norm = (X_val - mean) / (std + 1e-8)
    X_eval_norm = (X_eval - mean) / (std + 1e-8)
    
    # Tensorへの変換
    X_train_tensor = torch.FloatTensor(X_train_norm)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_norm)
    y_val_tensor = torch.LongTensor(y_val)
    X_eval_tensor = torch.FloatTensor(X_eval_norm)
    y_eval_tensor = torch.LongTensor(y_eval)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    myModel = gcn_select_net(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        transformer_dropout=transformer_dropout,
        cnn_dropout=cnn_dropout,
        gnn_dropout=gnn_dropout,
        num_attention_heads=num_attention_heads,
        gnn_heads=gnn_heads,
        cnn_out_channels=cnn_out_channels
    )
    myModel = myModel.to(device)
    
    optimizer = torch.optim.AdamW(myModel.parameters(), lr=learningRate, weight_decay=weight_decay)
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)
    else:
        scheduler = None
    
    y_train_list = y_train_tensor.tolist()
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_list), y=y_train_list)
    class_weights = torch.FloatTensor(weights).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_acc = 0.0
    best_val_bacc = 0.0
    best_val_combined_score = -float('inf')
    best_model_state = None
    best_val_true = []
    best_val_pred = []
    
    best_fold_attn_matrix = None      
    best_fold_elec_importance = None 
    
    early_stop_patience = 20
    epochs_no_improve = 0
    
    train_loss_history = np.zeros(epoch)
    val_loss_history = np.zeros(epoch)
    train_acc_history = np.zeros(epoch)
    val_acc_history = np.zeros(epoch)
    
    for i in range(epoch):
        ### 1. TRAIN ###
        myModel.train()
        total_train_loss = 0
        total_train_correct = 0
        train_step = 0

        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            outputs, _, _, _ = myModel(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            batch_correct = (preds == y_batch).sum().item()
            total_train_correct += batch_correct
            
            train_step += 1
            if train_step % 100 == 0:
                batch_acc = 100.0 * batch_correct / len(y_batch)
                print(f"  [Train] Epoch {i+1} | Step {train_step}/{len(train_dataloader)} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.2f}%")

        epoch_train_loss = total_train_loss / len(train_dataloader)
        epoch_train_acc = 100.0 * total_train_correct / len(train_dataset)

        if scheduler is not None:
            scheduler.step()

        ### 2. VAL ###
        myModel.eval()
        total_val_loss = 0
        total_val_correct = 0
        current_val_true = []
        current_val_pred = []
        
        current_epoch_attentions = [] 
        current_epoch_elec_imp = [] 

        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs, _, electrode_attention, gat_attention = myModel(x_batch)
                
                current_epoch_attentions.append(gat_attention.detach().cpu())
                current_epoch_elec_imp.append(electrode_attention.detach().cpu())
                
                loss = loss_fn(outputs, y_batch)
                total_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total_val_correct += (preds == y_batch).sum().item()
                
                current_val_true.extend(y_batch.cpu().tolist())
                current_val_pred.extend(preds.cpu().tolist())
                
        epoch_val_loss = total_val_loss / len(val_dataloader)
        epoch_val_acc = 100.0 * total_val_correct / len(val_dataset)
        epoch_val_bacc = balanced_accuracy_score(current_val_true, current_val_pred)
        epoch_val_kappa = cohen_kappa_score(current_val_true, current_val_pred)
        epoch_val_weighted_f1 = f1_score(current_val_true, current_val_pred, average='weighted', zero_division=0)
        
        epoch_val_combined_score = 0.4 * epoch_val_bacc + 0.3 * epoch_val_kappa + 0.3 * epoch_val_weighted_f1
        
        train_loss_history[i] = epoch_train_loss
        val_loss_history[i] = epoch_val_loss
        train_acc_history[i] = epoch_train_acc
        val_acc_history[i] = epoch_val_acc
        
        if epoch_val_combined_score > best_val_combined_score:
            best_val_combined_score = epoch_val_combined_score
            best_val_bacc = epoch_val_bacc
            best_val_acc = epoch_val_acc
            best_model_state = copy.deepcopy(myModel.state_dict())
            best_val_true = list(current_val_true)
            best_val_pred = list(current_val_pred)
            
            all_attn = torch.cat(current_epoch_attentions, dim=0)
            best_fold_attn_matrix = torch.mean(all_attn, dim=(0, 1)).numpy()
            
            all_elec = torch.cat(current_epoch_elec_imp, dim=1) 
            if all_elec.dim() == 3:
                best_fold_elec_importance = torch.mean(all_elec, dim=(0, 1))[1:].numpy()
            else:
                best_fold_elec_importance = torch.mean(all_elec, dim=0)[1:].numpy()
            
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        print(f"Epoch {i+1:03d}/{epoch} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | Val Comb: {epoch_val_combined_score:.4f} | No Improve: {epochs_no_improve}")
        
        if epochs_no_improve >= early_stop_patience:
            print(f"  Early stopping at epoch {i+1} due to no improvement for {early_stop_patience} epochs.")
            # グラフ描画時に末尾が急激に0に落ちないように、以降のエポックには最終値を埋める (Forward Fill)
            train_loss_history[i+1:] = epoch_train_loss
            val_loss_history[i+1:] = epoch_val_loss
            train_acc_history[i+1:] = epoch_train_acc
            val_acc_history[i+1:] = epoch_val_acc
            break
            
    best_val_kappa = cohen_kappa_score(best_val_true, best_val_pred)
    best_val_macro_recall = recall_score(best_val_true, best_val_pred, average='macro', zero_division=0)
    best_val_weighted_f1 = f1_score(best_val_true, best_val_pred, average='weighted', zero_division=0)
    print(f"Best Val Comb: {best_val_combined_score:.4f}, BACC: {best_val_bacc:.4f}, Kappa: {best_val_kappa:.4f}")
    
    # 描画と保存
    if save_results:
        if best_fold_attn_matrix is not None:
            visualize_attention(best_fold_attn_matrix, best_val_acc)
        if best_fold_elec_importance is not None:
            visualize_electrode_importance(best_fold_elec_importance, best_val_acc)
        
        try:
            cm_val = confusion_matrix(best_val_true, best_val_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_val)
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            ax.set_title(f'Best Val Confusion Matrix (Acc: {best_val_acc:.2f}%)')
            plt.savefig(f'{save_path}best_val_confusion_matrix.png')
            plt.close(fig) 
        except Exception as e:
            print(f"Error saving val confusion matrix: {e}")
            
        torch.save(best_model_state, f"{save_path}gcn_select_net_holdout.pth")
    
    eval_acc, eval_prec, eval_rec, eval_f1, eval_kap = 0.0, 0.0, 0.0, 0.0, 0.0
    fold_eval_true, fold_eval_pred = [], []

    # ----------------------------------------------------
    # FINAL TEST ON EVAL SET USING THIS BEST MODEL
    # ----------------------------------------------------
    if save_results:
        print(f"--- Testing best model on EVAL set ---")
        myModel.load_state_dict(best_model_state)
        myModel.eval()
        
        with torch.no_grad():
            for x_batch, y_batch in eval_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs, _, _, _ = myModel(x_batch)
                _, preds = torch.max(outputs, 1)
                fold_eval_true.extend(y_batch.cpu().tolist())
                fold_eval_pred.extend(preds.cpu().tolist())
                
        eval_acc = accuracy_score(fold_eval_true, fold_eval_pred) * 100.0
        eval_prec = precision_score(fold_eval_true, fold_eval_pred, average='macro', zero_division=0)
        eval_rec = recall_score(fold_eval_true, fold_eval_pred, average='macro', zero_division=0)
        eval_f1 = f1_score(fold_eval_true, fold_eval_pred, average='macro', zero_division=0)
        eval_kap = cohen_kappa_score(fold_eval_true, fold_eval_pred)
        
        print(f"[Result] Eval Results -> Acc: {eval_acc:.2f}%, F1: {eval_f1:.4f}, Kappa: {eval_kap:.4f}")
        
        try:
            cm = confusion_matrix(fold_eval_true, fold_eval_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues, ax=ax)
            ax.set_title(f'Eval Confusion Matrix (Acc: {eval_acc:.2f}%)')
            plt.savefig(f'{save_path}eval_confusion_matrix.png')
            plt.close(fig) 
        except Exception as e:
            print(f"Error saving eval confusion matrix: {e}")

    # 集計用の辞書を返す
    return {
        'best_val_acc': best_val_acc,
        'best_val_bacc': best_val_bacc,
        'best_val_kappa': best_val_kappa,
        'best_val_macro_recall': best_val_macro_recall,
        'best_val_weighted_f1': best_val_weighted_f1,
        'eval_acc': eval_acc,
        'eval_prec': eval_prec,
        'eval_rec': eval_rec,
        'eval_f1': eval_f1,
        'eval_kap': eval_kap,
        'eval_true': fold_eval_true,
        'eval_pred': fold_eval_pred,
        'best_fold_elec_importance': best_fold_elec_importance,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history
    }

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 2, 6),
        "transformer_dropout": trial.suggest_float("transformer_dropout", 0.1, 0.5),
        "cnn_dropout": trial.suggest_float("cnn_dropout", 0.1, 0.5),
        "gnn_dropout": trial.suggest_float("gnn_dropout", 0.1, 0.5),
        "num_attention_heads": trial.suggest_categorical("num_attention_heads", [8, 16]),
        "gnn_heads": trial.suggest_categorical("gnn_heads", [1, 2, 4]),
        "cnn_out_channels": trial.suggest_categorical("cnn_out_channels", [8, 16, 32]),
        "use_lr_scheduler": trial.suggest_categorical("use_lr_scheduler", [True, False]),
        "save_results": False
    }

    print(f"\n========== Starting Trial {trial.number} ==========")
    print("Trial Parameters:")
    for key, value in params.items():
        if key != 'save_results':
            print(f"  {key}: {value}")
    print("===================================================\n")

    X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes = load_data()
    
    res = run_trial(params, X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes, trial_number=trial.number)
    
    val_bacc = res['best_val_bacc']
    val_kappa = res['best_val_kappa']
    val_weighted_f1 = res['best_val_weighted_f1']
    
    combined_score = (0.4 * val_bacc) + (0.3 * val_kappa) + (0.3 * val_weighted_f1)
    
    print(f"Trial {trial.number} Finished -> Comb: {combined_score:.4f} (BACC: {val_bacc:.4f}, Kappa: {val_kappa:.4f}, W-F1: {val_weighted_f1:.4f})")
    
    return combined_score

def evaluate_with_best_params(best_params):
    best_params["save_results"] = True
    print("\n--- Training with Best Parameters ---")
    X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes = load_data()
    
    res = run_trial(best_params, X_train, y_train, X_val, y_val, X_eval, y_eval, num_classes)
    
    val_acc = res['best_val_acc']
    val_bacc = res['best_val_bacc']
    eval_acc = res['eval_acc']
    eval_prec = res['eval_prec']
    eval_rec = res['eval_rec']
    eval_f1 = res['eval_f1']
    eval_kap = res['eval_kap']
    
    all_eval_true_labels = res['eval_true']
    all_eval_pred_labels = res['eval_pred']
    
    elec_importance = res['best_fold_elec_importance']
    train_loss_history = res['train_loss_history']
    val_loss_history = res['val_loss_history']
    train_acc_history = res['train_acc_history']
    val_acc_history = res['val_acc_history']

    # ==========================================
    # Write final results to .txt & Visualize Globals
    # ==========================================
    print("\n===== Holdout Validation Finished =====")
    print(f"Val Accuracy:  {val_acc:.4f}%")
    print(f"Val BACC:      {val_bacc:.4f}")
    print(f"Eval Accuracy: {eval_acc:.4f}%")

    global_report = classification_report(all_eval_true_labels, all_eval_pred_labels, digits=4, zero_division=0)
    eval_bacc_global = balanced_accuracy_score(all_eval_true_labels, all_eval_pred_labels)

    with open(f'{save_path}holdout_eval_results.txt', 'w') as f:
        f.write("===== Holdout Validation Results =====\n")
        f.write(f"-> Val Accuracy: {val_acc:.4f}%\n")
        f.write(f"-> Val BACC:     {val_bacc:.4f}\n\n")
        
        f.write("===== Final Eval (Test) Results =====\n")
        f.write(f"-> Eval Accuracy:  {eval_acc:.4f}%\n")
        f.write(f"-> Eval BACC:      {eval_bacc_global:.4f}\n")
        f.write(f"-> Eval Precision: {eval_prec:.4f}\n")
        f.write(f"-> Eval Recall:    {eval_rec:.4f}\n")
        f.write(f"-> Eval F1-Score:  {eval_f1:.4f}\n")
        f.write(f"-> Eval Kappa:     {eval_kap:.4f}\n\n")
        
        if elec_importance is not None:
            sorted_indices = np.argsort(elec_importance)[::-1]
            sorted_channels = [(channel_names[idx], elec_importance[idx]) for idx in sorted_indices]
            f.write("===== Electrode Importance Ranking =====\n")
            for rank, (ch, score) in enumerate(sorted_channels, 1):
                f.write(f"{rank}. {ch}: {score:.6f}\n")
            f.write("\n")

        f.write("===== Eval Classification Report =====\n")
        f.write(global_report)

    try:
        cm_global = confusion_matrix(all_eval_true_labels, all_eval_pred_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_global)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(f'Eval Confusion Matrix (Acc: {eval_acc:.2f}%)')
        plt.savefig(f'{save_path}eval_confusion_matrix_overall.png')
        plt.close(fig)
    except Exception as e:
        print(f"Error saving global confusion matrix: {e}")

    # Global Electrode Importance Plot
    if elec_importance is not None:
        sorted_indices = np.argsort(elec_importance)[::-1]
        sorted_channels = [(channel_names[idx], elec_importance[idx]) for idx in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        ch_names_sorted = [ch for ch, _ in sorted_channels]
        scores_sorted = [score for _, score in sorted_channels]
        sns.barplot(x=ch_names_sorted, y=scores_sorted, palette="Reds_r")
        plt.title("Electrode Importance")
        plt.xlabel("Channel")
        plt.ylabel("Importance Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}electrode_importance_bar.png")
        plt.close()

        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2)
        ax.add_artist(circle)
        plt.plot([-0.1, 0, 0.1], [1.0, 1.1, 1.0], 'k-', linewidth=2)

        scores_array = np.array(scores_sorted)
        max_score = scores_array.max() if len(scores_array) > 0 else 1
        min_score = scores_array.min() if len(scores_array) > 0 else 0
        norm = plt.Normalize(vmin=min_score, vmax=max_score)
        try:
            cmap = plt.get_cmap("Reds")
        except AttributeError:
            cmap = plt.cm.Reds

        for ch, score in sorted_channels:
            x, y = coords.get(ch, (0, 0))
            color = cmap(norm(score))
            size = 1000 * ((score - min_score) / (max_score - min_score + 1e-9)) + 200
            plt.scatter(x, y, s=size, c=[color], edgecolors='black', zorder=10)
            plt.text(x, y, ch, ha='center', va='center', fontsize=7, fontweight='bold', color='black', zorder=11)
            
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Importance Score")

        plt.title("Spatial Distribution of Electrode Importance")
        plt.xlim(-1.2, 1.2); plt.ylim(-1.2, 1.2)
        plt.axis('off')
        plt.savefig(f"{save_path}electrode_importance_head.png")
        plt.close()

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss_history, label='Training Loss')
        plt.plot(epochs, val_loss_history, label='Validation Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        plt.savefig(f'{save_path}loss_curve.png'); plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc_history, label='Training Accuracy')
        plt.plot(epochs, val_acc_history, label='Validation Accuracy', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100) 
        plt.legend(); plt.grid(True)
        plt.savefig(f'{save_path}accuracy_curve.png'); plt.close()
        print("\nLearning curves saved.")
    except Exception as e:
        print(f"Plotting Error: {e}")

if __name__ == '__main__':
    # マルチプロセス時にCUDAエラーを防ぐための設定
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    set_seed(seed)
    
    num_gpus = torch.cuda.device_count()
    # メモリ消費が約750MBと少ないため、1つのGPUにつき複数のプロセスを起動
    processes_per_gpu = 5
    n_jobs = max(1, num_gpus * processes_per_gpu)
    
    print(f"Using {num_gpus} GPUs. Running {n_jobs} parallel trials ({processes_per_gpu} trials per GPU).")
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="TUEV_Hyperparameter_Optimization_Holdout_Expanded",
        storage="sqlite:///optuna_study_holdout_expanded.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=40, n_jobs=n_jobs)
    
    # 全試行のログ（各パラメータの組み合わせと結果）をCSVとして保存
    df = study.trials_dataframe()
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f"{save_path}optuna_trials_history.csv", index=False)
    print(f"\nSaved all trials history to '{save_path}optuna_trials_history.csv'")
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Evaluate and save results using the best hyperparameters
    evaluate_with_best_params(trial.params)