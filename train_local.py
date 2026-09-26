"""
DeepVerify - Local Trainer (production-grade)
=============================================
Trains a binary deepfake classifier on the local datasets/fake + datasets/real
folders and writes the result to app/deepfake_model.keras + app/model_meta.json.

Production hardening (vs. v1):
  * Early stopping on val_auc with best-weights restore (anti-overfitting).
  * ReduceLROnPlateau in fine-tune phase (smoother convergence).
  * Stronger augmentation + label smoothing.
  * Identity-grouped split: frames from one video stay in one fold (no leakage).
  * Per-epoch history dumped to app/training_history.json so the web UI can
    plot training trajectories.
  * Overfitting diagnostic printed at the end of each phase.

Output convention: model output = P(real). Threshold = 0.5.
"""

import json
import os
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Input,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation,
    Rescaling,
)
from tensorflow.keras.models import Model


# ----- Pure-numpy metrics (no sklearn dependency) ----------------------------
def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos, n_neg = len(pos), len(neg)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_ranks = sums / counts
    ranks = avg_ranks[inv]
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --- Config -------------------------------------------------------------------
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FAKE     = os.path.join(ROOT, "datasets", "fake")
DATA_REAL     = os.path.join(ROOT, "datasets", "real")
OUT_MODEL     = os.path.join(ROOT, "app", "deepfake_model.keras")
OUT_META      = os.path.join(ROOT, "app", "model_meta.json")
OUT_HISTORY   = os.path.join(ROOT, "app", "training_history.json")
INPUT_SIZE    = (224, 224)
FRAMES_PER_V  = 32
SEED          = 1337
EPOCHS_HEAD   = 14         # capped by EarlyStopping (patience=3)
EPOCHS_FT     = 12         # capped by EarlyStopping (patience=4)
BATCH_SIZE    = 32
LABEL_SMOOTH  = 0.05

np.random.seed(SEED)
tf.random.set_seed(SEED)


# --- Frame extraction ---------------------------------------------------------
def extract_frames(path, n=FRAMES_PER_V):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []
    n = min(n, total)
    indices = {round(i * (total - 1) / max(n - 1, 1)) for i in range(n)}
    out, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in indices:
            rgb = cv2.cvtColor(cv2.resize(frame, INPUT_SIZE), cv2.COLOR_BGR2RGB)
            out.append(rgb.astype(np.float32))
        idx += 1
    cap.release()
    return out


def collect(folder, label):
    """Identity-grouped split: each *video* goes entirely to train OR val."""
    videos = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
    )
    print(f"[data] {folder} -> {len(videos)} videos (label={label})")
    if not videos:
        raise RuntimeError(f"No videos found in {folder}")
    rng = np.random.RandomState(SEED + label)
    rng.shuffle(videos)
    n_val = max(1, int(round(len(videos) * 0.2)))
    val_v, train_v = videos[:n_val], videos[n_val:]

    def load(vlist):
        X, y = [], []
        for v in vlist:
            frames = extract_frames(v)
            print(f"  [{label}] {os.path.basename(v):60s} -> {len(frames)} frames")
            X.extend(frames); y.extend([label] * len(frames))
        return X, y

    Xtr, ytr = load(train_v)
    Xv,  yv  = load(val_v)
    return Xtr, ytr, Xv, yv


print("[stage] Extracting frames (identity-grouped split)...")
t0 = time.time()
# Label convention: 1 = real, 0 = fake. Model output = P(real).
Xtr_f, ytr_f, Xv_f, yv_f = collect(DATA_FAKE, 0)
Xtr_r, ytr_r, Xv_r, yv_r = collect(DATA_REAL, 1)

X_train = np.array(Xtr_f + Xtr_r, dtype=np.float32)
y_train = np.array(ytr_f + ytr_r, dtype=np.float32)
X_val   = np.array(Xv_f  + Xv_r,  dtype=np.float32)
y_val   = np.array(yv_f  + yv_r,  dtype=np.float32)

print(f"[data] train: {X_train.shape}  val: {X_val.shape}  ({time.time()-t0:.1f}s)")
print(f"[data] train fake/real: {int((1-y_train).sum())}/{int(y_train.sum())}")
print(f"[data] val   fake/real: {int((1-y_val).sum())}/{int(y_val.sum())}")

idx = np.arange(len(X_train)); np.random.RandomState(SEED).shuffle(idx)
X_train, y_train = X_train[idx], y_train[idx]


# --- Model --------------------------------------------------------------------
def build_model():
    """MobileNetV2 + transfer head with strong regularisation."""
    inp = Input(shape=(*INPUT_SIZE, 3))
    aug = RandomFlip("horizontal", seed=SEED)(inp)
    aug = RandomRotation(0.10, seed=SEED)(aug)
    aug = RandomZoom(0.12, seed=SEED)(aug)
    aug = RandomTranslation(0.06, 0.06, seed=SEED)(aug)
    aug = RandomContrast(0.18, seed=SEED)(aug)
    pre = Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess")(aug)

    base = MobileNetV2(input_shape=(*INPUT_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(pre, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.45)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.30)(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inp, out), base


print("[stage] Building model...")
model, backbone = build_model()
loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=loss_fn,
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

# Class weights (defensive even on balanced data)
n_pos = float(y_train.sum())
n_neg = float(len(y_train) - n_pos)
cw = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
print(f"[train] class_weight={cw}")


# --- Phase A: head only -------------------------------------------------------
print(f"[stage] Phase A: head-only training (max {EPOCHS_HEAD} epochs)")
cbA = [
    EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True, verbose=1),
]
hist_a = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_HEAD, batch_size=BATCH_SIZE,
    class_weight=cw, callbacks=cbA, verbose=2,
)


# --- Phase B: fine-tune top layers --------------------------------------------
print(f"[stage] Phase B: fine-tune top 30 layers (max {EPOCHS_FT} epochs)")
backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=loss_fn,
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
cbB = [
    EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
]
hist_b = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_FT, batch_size=BATCH_SIZE,
    class_weight=cw, callbacks=cbB, verbose=2,
)


# --- Overfitting diagnostic ---------------------------------------------------
def _gap(h, key):
    tr = h.history.get(key, [])
    va = h.history.get("val_" + key, [])
    if not tr or not va:
        return None
    return float(tr[-1] - va[-1])

gap_acc = _gap(hist_b, "accuracy") or _gap(hist_a, "accuracy") or 0.0
gap_loss = _gap(hist_b, "loss") or _gap(hist_a, "loss") or 0.0
print(f"[diag] Final train-val accuracy gap: {gap_acc*100:+.2f}%")
print(f"[diag] Final train-val loss gap   : {gap_loss:+.4f}")
if gap_acc > 0.15:
    print("[diag] WARNING: large generalisation gap (>15%) -> consider more data or stronger regularisation.")
else:
    print("[diag] OK: generalisation gap within healthy bounds.")


# --- Evaluate -----------------------------------------------------------------
print("[stage] Evaluating on validation set...")
p_real = model.predict(X_val, verbose=0).flatten()
y_real = y_val
auc = float(roc_auc_score(y_real, p_real))
acc = float(accuracy_score(y_real, (p_real >= 0.5).astype(int)))
print(f"[eval] P(real) AUC={auc:.4f}  ACC={acc:.4f}")


# --- Temperature calibration --------------------------------------------------
eps = 1e-7
logits = np.log((p_real + eps) / (1.0 - p_real + eps))


def nll(T):
    s = 1.0 / (1.0 + np.exp(-logits / T))
    s = np.clip(s, eps, 1 - eps)
    return -float(np.mean(y_real * np.log(s) + (1 - y_real) * np.log(1 - s)))


best_T, best_n = 1.0, nll(1.0)
for T in np.linspace(0.5, 3.0, 26):
    v = nll(float(T))
    if v < best_n:
        best_T, best_n = float(T), v
print(f"[calib] temperature={best_T:.3f}  nll={best_n:.4f}")


# --- Save model + meta + per-epoch history -----------------------------------
os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
model.save(OUT_MODEL)
print(f"[save] model -> {OUT_MODEL}")

meta = {
    "type": "trained",
    "architecture": "MobileNetV2 + transfer head (label-smoothed, identity-grouped split)",
    "version": "1.1.0-local",
    "accuracy": round(acc, 4),
    "auc": round(auc, 4),
    "temperature": round(best_T, 4),
    "frames_per_video": FRAMES_PER_V,
    "train_samples": int(len(X_train)),
    "val_samples": int(len(X_val)),
    "train_val_acc_gap": round(gap_acc, 4),
    "train_val_loss_gap": round(gap_loss, 4),
    "split_strategy": "identity-grouped (videos never share folds)",
    "label_smoothing": LABEL_SMOOTH,
    "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "note": "Trained locally on datasets/fake + datasets/real. Output = P(real).",
}
with open(OUT_META, "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"[save] meta  -> {OUT_META}")


def _series(key):
    return [float(v) for v in hist_a.history.get(key, [])] + [float(v) for v in hist_b.history.get(key, [])]


history_payload = {
    "epochs": list(range(1, len(_series("loss")) + 1)),
    "phase_break": len(hist_a.history.get("loss", [])),
    "loss":     _series("loss"),
    "val_loss": _series("val_loss"),
    "accuracy": _series("accuracy"),
    "val_accuracy": _series("val_accuracy"),
    "auc":     _series("auc"),
    "val_auc": _series("val_auc"),
    "final": {"acc": acc, "auc": auc, "temperature": best_T,
              "train_val_acc_gap": gap_acc, "train_val_loss_gap": gap_loss},
}
with open(OUT_HISTORY, "w") as fh:
    json.dump(history_payload, fh, indent=2)
print(f"[save] history -> {OUT_HISTORY}")
print("[done]")
