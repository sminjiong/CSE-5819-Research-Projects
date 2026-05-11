#%%
import pickle
from utils.dataset_cfg import SleepApnea
from data_utils.sleepapnea_dataset import SleepApneaDataset

root_dir = r"D:\vscode\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0\files"
cfg = SleepApnea(root_dir)
cfg.debug = False
cfg.use_spectrogram = True

train_dataset = SleepApneaDataset(root_dir, cfg.train_set, cfg)
val_dataset   = SleepApneaDataset(root_dir, cfg.val_set, cfg)
test_dataset  = SleepApneaDataset(root_dir, cfg.eval_set, cfg)

with open("dataset_cache.pkl", "wb") as f:
    pickle.dump((train_dataset, val_dataset, test_dataset), f)
print("✅ 저장 완료!")






#%%
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score
from utils.dataset_cfg import SleepApnea
from models.our_models import CrossAttnTransformerClf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cfg
root_dir = r"D:\vscode\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0\files"
cfg = SleepApnea(root_dir)
cfg.debug = False
cfg.use_spectrogram = True

# 데이터 로드
with open("dataset_cache.pkl", "rb") as f:
    train_dataset, val_dataset, test_dataset = pickle.load(f)
print(f"✅ 로드 완료! Train:{len(train_dataset)} Val:{len(val_dataset)} Test:{len(test_dataset)}")

# DataLoader
val_loader  = DataLoader(val_dataset,  batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)








#%% Model
# ---------------------------
model = CrossAttnTransformerClf(
    cfg=cfg,
    num_classes=2,
    input_length=cfg.input_length,  # 3000 (30sec * 100Hz)
    d_model=64,
    nhead=4,
    num_layers_per_modal=1,   # ✅ 1로 줄임 (다운샘플링 과도 방지)
    num_layers=2,
    dropout=0.1,
    verbose=False,
    base_factor=3,            # ✅ 5→3으로 줄임 (안정성)
    num_experts=2             # ✅ 4→2로 줄임 (데이터 적으니)
).to(device)


# ---------------------------
# 🔥 Class imbalance 처리 (중복 제거됨)
# ---------------------------
labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]

pos = sum(labels)
neg = len(labels) - pos

print(f"\n🔥 Class distribution → Normal: {neg}, Apnea: {pos}")
print(f"🔥 Apnea ratio: {pos/len(labels):.4f}")

from sklearn.utils.class_weight import compute_class_weight

# labels_np = np.array(labels)
from torch.utils.data import WeightedRandomSampler

labels_np = np.array(labels)
class_counts = np.bincount(labels_np)
sample_weights = 1.0 / class_counts[labels_np]

# ✅ Sampler만 사용
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float32),
    num_samples=len(sample_weights),
    replacement=True
)
train_loader = DataLoader(
    train_dataset, batch_size=64,
    sampler=sampler,
    num_workers=0
)

# ✅ weight 없는 CrossEntropy
criterion = nn.CrossEntropyLoss()

# ✅ lr은 1e-4 유지 (좋음)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.75, gamma=2.0):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, logits, targets):
#         ce = nn.functional.cross_entropy(logits, targets, reduction='none')
#         pt = torch.exp(-ce)
#         focal = self.alpha * (1 - pt) ** self.gamma * ce
#         return focal.mean()

# criterion = FocalLoss(alpha=0.75, gamma=2.0)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# Helper
# ---------------------------
def prepare_input(batch):
    x = torch.stack([
        batch['ECG'],
        batch['SpO2'],
        batch['sound']
    ], dim=2)
    return x.to(device)

# ---------------------------
# 🔥 Evaluation (통일)
# ---------------------------
def evaluate(model, loader, threshold):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch, label in loader:

            x = prepare_input(batch)
            y = label.to(device)

            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob > threshold).long()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)
    # print(np.unique(y_pred, return_counts=True))

    f1      = f1_score(y_true, y_pred)
    recall  = recall_score(y_true, y_pred)   # 무호흡 탐지율
    precision = precision_score(y_true, y_pred)

    print(f"Recall(Sensitivity): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1: {f1:.4f}")

    return acc, bal_acc, f1, recall, precision


# ---------------------------
# 🔥 Threshold 자동 탐색
# ---------------------------
def find_best_threshold(model, loader):

    best_score = -1
    best_t = 0.5

    # 0.02, 0.20, 19
    for t in np.linspace(0.2, 0.8, 13):

        acc, bal_acc, f1, recall, precision = evaluate(model, loader, threshold=t)
        if f1 == 0:
            continue
        score = 0.3 * bal_acc + 0.7 * f1

        print(f"t={t:.2f} | acc={acc:.4f} | f1={f1:.4f} | score={score:.4f}")

        if score > best_score:
            best_score = score
            best_t = t

    print(f"🔥 Best threshold: {best_t:.2f}")
    return best_t













#%% Training
# ---------------------------
best_val_f1 = 0
best_val_score = 0
best_t_saved = 0.5

for epoch in range(10):
    import time
   
    start = time.time()
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, (batch, label) in enumerate(pbar):

        x = prepare_input(batch)
        y = label.to(device)

        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

    total_loss /= len(train_dataset)

    # validation
    # ✅ 매 epoch마다 threshold 탐색하지 않고
    # 5 epoch마다 한 번만 탐색
    if epoch % 3 == 0 or epoch == 9:
        best_t = find_best_threshold(model, val_loader)
    else:
        best_t = best_t_saved  # 이전 best threshold 재사용

    val_acc, val_bal_acc, val_f1, val_recall, val_precision = evaluate(model, val_loader, threshold=best_t)


    
    print(f"\nEpoch {epoch} | Loss {total_loss:.4f} | Val Acc {val_acc:.4f} | Bal Acc {val_bal_acc:.4f}")
    print(f"\nEpoch {epoch} | Val Recall {val_recall:.4f} | Val precision {val_precision:.4f} | Val F1 {val_f1:.4f}")
    print(f"⏱ Epoch time: {time.time() - start:.2f} sec")

    val_score = 0.3 * val_bal_acc + 0.7 * val_f1

    if val_score > best_val_score:
        best_val_score = val_score
        best_t_saved = best_t
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Best model saved!")


# ---------------------------
# 🔥 Best model 평가
# ---------------------------
print("\n🚀 Loading best model...")
model.load_state_dict(torch.load("best_model.pth"))

# threshold 찾기

# 최종 test
# test_acc, test_f1 = evaluate(model, test_loader, threshold=best_t)
test_acc, test_bal_acc, test_f1, test_recall, test_precision= evaluate(model, test_loader, threshold=best_t_saved)

print(f"\n🔥 Final Test Acc: {test_acc:.4f} and Test Bal Acc: {test_bal_acc:.4f}")
print(f"🔥 Final Test F1: {test_f1:.4f}")
print(f"🔥 Final Test precision: {test_precision:.4f}")
print(f"🔥 Final Test recall: {test_recall:.4f}")











#%% visual aids
#%% visual aids
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------
# 1. 전체 확률 분포 수집
# ---------------------------
def get_probs_and_labels(model, loader):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for batch, label in loader:
            x = prepare_input(batch)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            prob = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(label.numpy())
            y_score.extend(prob.cpu().numpy())
    return np.array(y_true), np.array(y_score)

y_true, y_score = get_probs_and_labels(model, test_loader)
y_pred = (y_score > best_t_saved).astype(int)

# ---------------------------
# 2. Classification Report
# ---------------------------
print("\n📊 Classification Report")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Apnea']))

# ---------------------------
# 3. 핵심 지표 출력
# ---------------------------
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, precision_score
bal_acc   = balanced_accuracy_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc   = auc(fpr, tpr)

print(f"\n{'='*40}")
print(f"Balanced Accuracy : {bal_acc:.4f}")
print(f"F1 Score          : {f1:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"AUC               : {roc_auc:.4f}")
print(f"Best Threshold    : {best_t_saved:.2f}")
print(f"{'='*40}")

# ---------------------------
# 4. 종합 시각화 (2x3 grid)
# ---------------------------
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Sleep Apnea Detection — Evaluation Results", fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# --- (1) ROC Curve ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(fpr, tpr, color='steelblue', lw=2, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], 'k--', lw=1)
ax1.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve")
ax1.legend()
ax1.grid(alpha=0.3)

# --- (2) Confusion Matrix ---
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Apnea'])
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title("Confusion Matrix")

# --- (3) F1 vs Threshold ---
ax3 = fig.add_subplot(gs[0, 2])
thresholds = np.linspace(0.1, 0.9, 17)
f1_list, recall_list, precision_list = [], [], []
for t in thresholds:
    pred_t = (y_score > t).astype(int)
    f1_list.append(f1_score(y_true, pred_t, zero_division=0))
    recall_list.append(recall_score(y_true, pred_t, zero_division=0))
    precision_list.append(precision_score(y_true, pred_t, zero_division=0))

ax3.plot(thresholds, f1_list,        label='F1',        color='steelblue',  lw=2)
ax3.plot(thresholds, recall_list,    label='Recall',    color='coral',      lw=2, linestyle='--')
ax3.plot(thresholds, precision_list, label='Precision', color='mediumseagreen', lw=2, linestyle=':')
ax3.axvline(best_t_saved, color='gray', linestyle='--', lw=1.5, label=f'Best t={best_t_saved:.2f}')
ax3.set_xlabel("Threshold")
ax3.set_ylabel("Score")
ax3.set_title("Metrics vs Threshold")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# --- (4) Probability Distribution ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(y_score[y_true == 0], bins=40, alpha=0.6, color='steelblue', label='Normal', density=True)
ax4.hist(y_score[y_true == 1], bins=40, alpha=0.6, color='coral',     label='Apnea',  density=True)
ax4.axvline(best_t_saved, color='gray', linestyle='--', lw=1.5, label=f'threshold={best_t_saved:.2f}')
ax4.set_xlabel("Predicted Probability (Apnea)")
ax4.set_ylabel("Density")
ax4.set_title("Probability Distribution")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# --- (5) Bar chart: 핵심 지표 비교 ---
ax5 = fig.add_subplot(gs[1, 1])
metrics = ['Bal. Acc', 'F1', 'Recall', 'Precision', 'AUC']
values  = [bal_acc, f1, recall, precision, roc_auc]
colors  = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'goldenrod']
bars = ax5.bar(metrics, values, color=colors, alpha=0.8, edgecolor='white')
ax5.set_ylim(0, 1.0)
ax5.axhline(0.5, color='gray', linestyle='--', lw=1, label='Random baseline')
for bar, val in zip(bars, values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax5.set_title("Key Metrics Summary")
ax5.set_ylabel("Score")
ax5.legend(fontsize=8)
ax5.grid(axis='y', alpha=0.3)

# --- (6) Precision-Recall Curve ---
from sklearn.metrics import precision_recall_curve, average_precision_score
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(rec_curve, prec_curve, color='mediumpurple', lw=2, label=f'AP = {ap:.3f}')
ax6.axhline(y_true.mean(), color='gray', linestyle='--', lw=1, label=f'Baseline ({y_true.mean():.2f})')
ax6.fill_between(rec_curve, prec_curve, alpha=0.1, color='mediumpurple')
ax6.set_xlabel("Recall")
ax6.set_ylabel("Precision")
ax6.set_title("Precision-Recall Curve")
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

plt.savefig("evaluation_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ 저장 완료: evaluation_results.png")
