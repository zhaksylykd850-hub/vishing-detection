import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
import joblib

import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────
# GPU КОНФИГУРАЦИЯ
# ─────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Қолданылатын құрылғы: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 0. КОНФИГУРАЦИЯ
# ─────────────────────────────────────────
INPUT_PATH   = "text_label_preprocessed.csv"
OUTPUT_DIR   = "results_new2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
MAX_FEATURES = 10000
FRAUD_LABEL  = "fraud"
NORMAL_LABEL = "normal"
RUBERT_MODEL = "DeepPavlov/rubert-base-cased-sentence"

# ─────────────────────────────────────────
# 1. ДЕРЕКТЕРДІ ОҚУ
# ─────────────────────────────────────────
print("=" * 60)
print("1. Деректерді жүктеу...")
df = pd.read_csv(INPUT_PATH)
df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
print(f"   Жалпы жол саны : {len(df)}")
print(f"   Лейбл үлестірімі:\n{df['label'].value_counts()}")

label_map = {FRAUD_LABEL: 1, NORMAL_LABEL: 0}
df["label_enc"] = df["label"].map(label_map)
fraud_code  = 1
normal_code = 0
print(f"   Кодировка: fraud={fraud_code}, normal={normal_code}")

X = df["text"].values
y = df["label_enc"].values

# ─────────────────────────────────────────
# 2. TRAIN / TEST БӨЛУ
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y
)
print(f"\n2. Train/Test бөлу:")
print(f"   Train: {len(X_train)} жол | Test: {len(X_test)} жол")
print(f"   Train fraud саны : {(y_train == 1).sum()}")
print(f"   Train normal саны: {(y_train == 0).sum()}")

n_fraud  = (y_train == 1).sum()
n_normal = (y_train == 0).sum()

# ─────────────────────────────────────────
# 3. КЛАСС САЛМАҚТАРЫ
# ─────────────────────────────────────────
class_weight_dict = {
    0: len(y_train) / (2 * n_normal),
    1: len(y_train) / (2 * n_fraud)
}
print(f"\n3. Класс салмақтары:")
print(f"   fraud  саны: {n_fraud}  → салмақ: {class_weight_dict[1]:.2f}")
print(f"   normal саны: {n_normal} → салмақ: {class_weight_dict[0]:.2f}")

# ─────────────────────────────────────────
# 4. ruBERT ЭМБЕДДИНГТЕРІ
# ─────────────────────────────────────────
print(f"\n4. ruBERT эмбеддингтері жасалуда...")
print(f"   Модель: {RUBERT_MODEL}")
print(f"   Құрылғы: CPU (RTX 5050 sm_120 үйлеспеу)")
print(f"   ⏳ ~2-5 минут күт...")

from sentence_transformers import SentenceTransformer
rubert = SentenceTransformer(RUBERT_MODEL, device="cpu")

X_train_emb = rubert.encode(
    X_train.tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)
X_test_emb = rubert.encode(
    X_test.tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"   Эмбеддинг өлшемі: {X_train_emb.shape[1]} измерение")
np.save(os.path.join(OUTPUT_DIR, "train_embeddings.npy"), X_train_emb)
np.save(os.path.join(OUTPUT_DIR, "test_embeddings.npy"),  X_test_emb)
joblib.dump(rubert, os.path.join(OUTPUT_DIR, "rubert_encoder.pkl"))
print("   ruBERT эмбеддингтері сақталды.")

# ─────────────────────────────────────────
# 5. CNN-LSTM (optimized) үшін KERAS TOKENIZER
# ─────────────────────────────────────────
print("\n5. Keras Tokenizer дайындалуда...")

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"   ✅ TensorFlow GPU: {len(gpus)} GPU")
else:
    print("   ⚠️  TensorFlow CPU қолданылады")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D,
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

VOCAB_SIZE = 15000
MAX_LEN    = 100
BATCH_SIZE = 32
EPOCHS     = 30

keras_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
keras_tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(
    keras_tokenizer.texts_to_sequences(X_train),
    maxlen=MAX_LEN, padding="post"
)
X_test_seq = pad_sequences(
    keras_tokenizer.texts_to_sequences(X_test),
    maxlen=MAX_LEN, padding="post"
)
joblib.dump(keras_tokenizer, os.path.join(OUTPUT_DIR, "keras_tokenizer.pkl"))
print("   Keras Tokenizer дайын.")

# ─────────────────────────────────────────
# 6. МОДЕЛЬДЕР КОНФИГУРАЦИЯСЫ
# ─────────────────────────────────────────
COLORS = {
    "NB + BERT":            "#1565C0",
    "SBERT + Ensemble":     "#00695C",
    "CNN-LSTM (optimized)": "#6A1B9A",
}

SHORT_NAMES = {
    "NB + BERT":            "NB + BERT",
    "SBERT + Ensemble":     "SBERT + Ensemble",
    "CNN-LSTM (optimized)": "CNN-LSTM (optimized)",
}

# ─────────────────────────────────────────
# 7. EVALUATE ФУНКЦИЯСЫ
# ─────────────────────────────────────────
def evaluate(name, y_true, y_pred, y_prob, color):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    short = SHORT_NAMES.get(name, name)
    print(f"\n   {'─'*45}")
    print(f"   {short}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}  (fraud класы)")
    print(f"   Recall   : {rec:.4f}  (fraud класы)")
    print(f"   F1-score : {f1:.4f}  (fraud класы)")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=[NORMAL_LABEL, FRAUD_LABEL],
        zero_division=0
    ))

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[NORMAL_LABEL, FRAUD_LABEL],
        yticklabels=[NORMAL_LABEL, FRAUD_LABEL],
        ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title(f"Confusion Matrix\n{short}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Болжам (Predicted)", fontsize=10)
    ax.set_ylabel("Нақты (Actual)", fontsize=10)
    plt.tight_layout()
    safe = short.replace(" ", "_").replace("+", "").replace("\n", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{safe}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc,
        "CM": cm, "color": color
    }

# ─────────────────────────────────────────
# 8. МОДЕЛЬДЕРДІ ОҚЫТУ
# ─────────────────────────────────────────
print("\n6. Модельдерді оқыту...")
results = {}

# ── 8a. NB + BERT (Oyeyemi 2023 — NB+BERT 97.31%) ──
print("\n   Оқытылуда: NB + BERT...")
nb_bert = GaussianNB()
nb_bert.fit(X_train_emb, y_train)
y_pred_nb = nb_bert.predict(X_test_emb)
y_prob_nb = nb_bert.predict_proba(X_test_emb)[:, 1]
results["NB + BERT"] = evaluate(
    "NB + BERT", y_test, y_pred_nb, y_prob_nb, COLORS["NB + BERT"]
)
results["NB + BERT"]["model"] = nb_bert
joblib.dump(nb_bert, os.path.join(OUTPUT_DIR, "model_NB_BERT.pkl"))

# ── 8b. ruBERT + Ensemble ──
print("\n   Оқытылуда: ruBERT + Ensemble...")
lr_clf  = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
)
rf_clf  = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
svc_clf = CalibratedClassifierCV(
    LinearSVC(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
)
ensemble = VotingClassifier(
    estimators=[("lr", lr_clf), ("rf", rf_clf), ("svc", svc_clf)],
    voting="soft"
)
ensemble.fit(X_train_emb, y_train)
y_pred_ens = ensemble.predict(X_test_emb)
y_prob_ens = ensemble.predict_proba(X_test_emb)[:, 1]
results["ruBERT + Ensemble"] = evaluate(
    "ruBERT + Ensemble", y_test, y_pred_ens, y_prob_ens, COLORS["ruBERT + Ensemble"]
)
results["ruBERT + Ensemble"]["model"] = ensemble
joblib.dump(ensemble, os.path.join(OUTPUT_DIR, "model_SBERT_Ensemble.pkl"))

# ── 8c. CNN-LSTM (optimized) (Al Saidat 2024) ──
print("\n   Оқытылуда: CNN-LSTM (optimized)...")

def build_cnn_lstm_optimized():
    model = Sequential([
        Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN),
        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

cnn_lstm_opt = build_cnn_lstm_optimized()
cnn_lstm_opt.summary()

history_opt = cnn_lstm_opt.fit(
    X_train_seq, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(
        monitor="val_loss", patience=5,
        restore_best_weights=True, verbose=1
    )],
    verbose=1
)

# CNN-LSTM (optimized) тарихы графигі
fig_hist, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_opt.history["loss"],     label="Train Loss", color="#6A1B9A")
axes[0].plot(history_opt.history["val_loss"], label="Val Loss",   color="#CE93D8", linestyle="--")
axes[0].set_title("CNN-LSTM (optimized): Loss тарихы", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history_opt.history["accuracy"],     label="Train Acc", color="#1B5E20")
axes[1].plot(history_opt.history["val_accuracy"], label="Val Acc",   color="#66BB6A", linestyle="--")
axes[1].set_title("CNN-LSTM (optimized): Accuracy тарихы", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cnn_lstm_optimized_history.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   CNN-LSTM (optimized) тарихы сақталды.")

y_prob_opt = cnn_lstm_opt.predict(X_test_seq, verbose=0).flatten()
y_pred_opt = (y_prob_opt >= 0.5).astype(int)
results["CNN-LSTM (optimized)"] = evaluate(
    "CNN-LSTM (optimized)", y_test, y_pred_opt, y_prob_opt, COLORS["CNN-LSTM (optimized)"]
)
results["CNN-LSTM (optimized)"]["model"] = cnn_lstm_opt
cnn_lstm_opt.save(os.path.join(OUTPUT_DIR, "model_CNN_LSTM_optimized.keras"))

# ─────────────────────────────────────────
# 9. САЛЫСТЫРМАЛЫ МЕТРИКА ГРАФИКТЕРІ
# ─────────────────────────────────────────
print("\n7. Салыстырмалы графиктер жасалуда...")

short_labels = [SHORT_NAMES.get(k, k) for k in results.keys()]
metric_keys  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
colors_list  = [v["color"] for v in results.values()]

fig = plt.figure(figsize=(20, 11))
fig.suptitle(
    "3 Модельдің Салыстырмалы Метрика Нәтижелері",
    fontsize=16, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
metric_positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]

for idx, metric in enumerate(metric_keys):
    r, c = metric_positions[idx]
    ax = fig.add_subplot(gs[r, c])
    vals = [results[k][metric] for k in results.keys()]
    bars = ax.bar(
        short_labels, vals, color=colors_list,
        edgecolor="white", linewidth=1.2, width=0.45
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1A1A1A"
        )
    ax.set_title(metric, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Мән", fontsize=9)
    ax.set_xticklabels(short_labels, rotation=10, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Жиынтық кесте — Precision, Recall қосылды
ax_table = fig.add_subplot(gs[1, 2])
ax_table.axis("off")

table_data = []
for k in results.keys():
    rv = results[k]
    table_data.append([
        SHORT_NAMES.get(k, k),
        f"{rv['Accuracy']:.3f}",
        f"{rv['Precision']:.3f}",
        f"{rv['Recall']:.3f}",
        f"{rv['F1']:.3f}",
        f"{rv['ROC-AUC']:.3f}"
    ])

tbl = ax_table.table(
    cellText=table_data,
    colLabels=["Модель", "Acc", "Prec", "Rec", "F1", "AUC"],
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.8)

for j in range(6):
    tbl[(0, j)].set_facecolor("#1E3A5F")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

best_key = max(results, key=lambda k: results[k]["F1"])
best_idx = list(results.keys()).index(best_key)
for j in range(6):
    tbl[(best_idx + 1, j)].set_facecolor("#E8F5E9")
    tbl[(best_idx + 1, j)].set_text_props(fontweight="bold", color="#1B5E20")

ax_table.set_title("Жиынтық нәтижелер\n(жасыл — үздік)", fontsize=9, fontweight="bold")

plt.savefig(os.path.join(OUTPUT_DIR, "comparison_all_metrics.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   Салыстырмалы график сақталды.")

# ─────────────────────────────────────────
# 10. ACCURACY САЛЫСТЫРУ ГРАФИГІ
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
acc_vals = [results[k]["Accuracy"] for k in results.keys()]
bars = ax.barh(
    short_labels[::-1], acc_vals[::-1],
    color=colors_list[::-1],
    edgecolor="white", linewidth=1.5, height=0.45
)
for bar, val in zip(bars, acc_vals[::-1]):
    ax.text(
        val + 0.003, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", ha="left",
        fontsize=12, fontweight="bold", color="#1A1A1A"
    )
ax.set_xlim(0, 1.12)
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_title("3 Модельдің Accuracy Салыстыруы", fontsize=14, fontweight="bold", pad=12)
ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.4, label="0.90 шек")
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   Accuracy салыстыру графигі сақталды.")

# ─────────────────────────────────────────
# 11. ЕҢ ТИІМДІ МОДЕЛЬДІ САҚТАУ
# ─────────────────────────────────────────
print("\n8. Үздік модель анықталуда...")
best_key   = max(results, key=lambda k: results[k]["F1"])
best_short = SHORT_NAMES.get(best_key, best_key)
best_f1    = results[best_key]["F1"]
best_acc   = results[best_key]["Accuracy"]
best_auc   = results[best_key]["ROC-AUC"]

print(f"\n{'='*60}")
print(f"   ✅ ЕҢ ТИІМДІ МОДЕЛЬ: {best_short}")
print(f"   F1-score  : {best_f1:.4f}")
print(f"   Accuracy  : {best_acc:.4f}")
print(f"   ROC-AUC   : {best_auc:.4f}")
print(f"{'='*60}")

if best_key == "CNN-LSTM (optimized)":
    results[best_key]["model"].save(os.path.join(OUTPUT_DIR, "best_model.keras"))
    joblib.dump(keras_tokenizer, os.path.join(OUTPUT_DIR, "best_model_tokenizer.pkl"))
    print("   Сақталды: best_model.keras + best_model_tokenizer.pkl")
else:
    # NB + BERT немесе SBERT + Ensemble
    joblib.dump(results[best_key]["model"], os.path.join(OUTPUT_DIR, "best_model.pkl"))
    joblib.dump(rubert, os.path.join(OUTPUT_DIR, "best_model_rubert.pkl"))
    print("   Сақталды: best_model.pkl + best_model_rubert.pkl")

import json
meta = {
    "best_model":  best_short,
    "f1_score":    round(best_f1, 4),
    "accuracy":    round(best_acc, 4),
    "roc_auc":     round(best_auc, 4),
    "rubert_model": RUBERT_MODEL,
    "all_results": {
        SHORT_NAMES.get(k, k): {
            "Accuracy":  round(v["Accuracy"],  4),
            "Precision": round(v["Precision"], 4),
            "Recall":    round(v["Recall"],    4),
            "F1":        round(v["F1"],        4),
            "ROC-AUC":   round(v["ROC-AUC"],   4)
        }
        for k, v in results.items()
    }
}
with open(os.path.join(OUTPUT_DIR, "results_summary.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n9. Барлық файлдар сақталды:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"   📁 {fname:50s} {size/1024:8.1f} KB")

print("\n✅ Оқыту аяқталды!")