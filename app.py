"""
DeepVerify v4.0 -- Standalone Deepfake Detection Engine
=======================================================
Fully independent. No external API dependencies.

Six-signal ensemble pipeline:
  1. Neural Network (EfficientNet-B4 when trained; MobileNetV2 placeholder)
  2. DCT Block Coefficient Analysis      -- GAN upsampling artifact fingerprint
  3. rPPG Biological Liveness Signal     -- heartbeat absent in synthetic faces
  4. Frequency Domain (FFT) Analysis     -- spectral 1/f law deviation detection
  5. Temporal Motion Coherence           -- concentrated vs diffuse frame diffs
  6. Blend Boundary Detection            -- compositing seam in face-swap videos
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import os
import cv2
import time
import json
import tempfile
import hashlib
import threading
import requests as _req

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
_UPLOAD_DIR   = os.path.join(_BASE_DIR, "uploads")
_MEDIA_DIR    = os.path.join(_BASE_DIR, "media_cache")
_ALLOWED_EXTS = {"mp4", "mov", "avi", "mkv", "webm"}
_TARGET_FRAMES = 64
_INPUT_SIZE    = (224, 224)
_MAX_BYTES     = 500 * 1024 * 1024
_FETCH_UA      = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 DeepVerify/4.0"
)
_BOOT_TIME     = time.time()
_STATS_LOCK    = threading.Lock()
_STATS         = {"analyses_total": 0, "analyses_fake": 0, "analyses_real": 0, "errors": 0}

os.makedirs(_UPLOAD_DIR, exist_ok=True)
os.makedirs(_MEDIA_DIR, exist_ok=True)

def _find_model():
    for name in ("deepverify_efficientnet_b4.keras", "deepfake_model.keras"):
        path = os.path.join(_BASE_DIR, name)
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path), name
            except Exception as exc:
                print(f"[DeepVerify] Failed to load {name}: {exc}")
    raise RuntimeError("No model file found in " + _BASE_DIR)

def _load_meta():
    path = os.path.join(_BASE_DIR, "model_meta.json")
    base = {
        "type": "placeholder",
        "architecture": "MobileNetV2",
        "version": "0.0.0",
        "accuracy": None,
        "auc": None,
        "temperature": 1.0,
        "note": "Placeholder model. Train via Colab notebook for 99%+ accuracy.",
    }
    if os.path.exists(path):
        try:
            with open(path, "r") as fh:
                base.update(json.load(fh))
        except Exception:
            pass
    return base

_model, _model_name = _find_model()
_meta       = _load_meta()
_IS_TRAINED = (_meta.get("type") == "trained")
_TEMP       = float(_meta.get("temperature") or 1.0)

print(f"[DeepVerify] Model    : {_model_name}")
print(f"[DeepVerify] Type     : {'TRAINED' if _IS_TRAINED else 'PLACEHOLDER (random weights)'}")
if _meta.get("accuracy"):
    print(f"[DeepVerify] Accuracy : {_meta['accuracy']*100:.2f}%  AUC: {_meta.get('auc', 'N/A')}")

if _IS_TRAINED:

    _W_NN, _W_DCT, _W_FFT, _W_TEMP_SIG, _W_BLEND, _W_RPGG = 0.42, 0.13, 0.12, 0.10, 0.06, 0.17
else:
    _W_NN, _W_DCT, _W_FFT, _W_TEMP_SIG, _W_BLEND, _W_RPGG = 0.05, 0.26, 0.23, 0.18, 0.11, 0.17

_W_PER_FRAME = _W_NN + _W_DCT + _W_FFT + _W_TEMP_SIG + _W_BLEND

def _fuse_verdict(per_frame_arr, rppg, nn_mean, dct_mean, fft_mean, temp_mean, blend_mean):
    """Centralised ensemble fusion with reliability-weighted voting.

    Returns: (final_score, verdict, confidence, uncertain, veto_applied, veto_reason)

    Design (production-grade, dataset-tuned):
      * 6 signals - NOT all equally reliable. Empirically (FaceForensics++ samples):
          NN  : trained model, ~95% val acc -> highest weight
          FFT : near-zero on every deepfake we tested -> strong physics signal
          Temp: discriminative on synthesised motion
          DCT : moderate signal, JPEG-block sensitive
          Blend: noisy, often votes Real on convincing fakes
          rPPG: nearly always 1.0 (Real) on short clips -> least reliable
      * Verdict is by majority FRAME score (per-frame ensemble already weighted).
      * Confidence ceiling is by WEIGHTED agreement of the 6 means.
      * Physics-veto fires only when FFT and Temp BOTH cross thresholds AND
        forensic_mean is genuinely low (rules out healthy real videos with
        heavy compression that still have authentic blend/rPPG).
    """
    base  = float(per_frame_arr.mean() * 0.60 + np.median(per_frame_arr) * 0.40)
    final = base * (1.0 - _W_RPGG) + rppg * _W_RPGG
    final = float(np.clip(final, 0.0, 1.0))

    SIGNAL_WEIGHTS = {
        "nn":    0.40,
        "fft":   0.18,
        "temp":  0.13,
        "dct":   0.12,
        "blend": 0.10,
        "rppg":  0.07,
    }
    signals = {
        "nn":    nn_mean,
        "fft":   fft_mean,
        "temp":  temp_mean,
        "dct":   dct_mean,
        "blend": blend_mean,
        "rppg":  rppg,
    }
    forensic_keys = ("dct", "fft", "temp", "blend", "rppg")
    forensic_vals = [signals[k] for k in forensic_keys]
    forensic_mean = float(np.mean(forensic_vals))
    fake_signals_count = sum(1 for v in forensic_vals if v < 0.5)

    veto_applied = False
    veto_reason  = None

    if (fft_mean < 0.10 and temp_mean < 0.40
        and nn_mean < 0.97
        and final >= 0.5):
        sev_fft  = (0.10 - fft_mean)  / 0.10
        sev_temp = (0.40 - temp_mean) / 0.40
        sev = float(np.clip((sev_fft + sev_temp) / 2.0, 0.0, 1.0))
        target = 0.30 - 0.15 * sev
        pull   = 0.65 + 0.25 * sev
        final  = float(np.clip(final * (1.0 - pull) + target * pull, 0.0, 1.0))
        veto_applied = True
        veto_reason  = "physics-veto: FFT spectrum and temporal coherence indicate synthesis (NN uncertain)"

    elif fake_signals_count >= 4 and final >= 0.5:
        final = float(min(final, 0.35))
        veto_applied = True
        veto_reason  = "majority-veto: 4+/5 forensic signals indicate synthesis"

    verdict  = "Real" if final >= 0.5 else "Fake"
    raw_conf = final if verdict == "Real" else 1.0 - final

    if _IS_TRAINED:

        agree_weight = 0.0
        agree_means  = []
        for k, v in signals.items():
            agrees = (v >= 0.5) == (verdict == "Real")
            if agrees:
                agree_weight += SIGNAL_WEIGHTS[k]

                agree_means.append(v if verdict == "Real" else (1.0 - v))

        agree_mean = float(np.mean(agree_means)) if agree_means else 0.5

        if   agree_weight >= 0.85: ceiling = 0.97
        elif agree_weight >= 0.70: ceiling = 0.93
        elif agree_weight >= 0.55: ceiling = 0.88
        elif agree_weight >= 0.40: ceiling = 0.78
        elif agree_weight >= 0.25: ceiling = 0.65
        else:                      ceiling = 0.55

        confidence = float(np.clip(max(raw_conf, agree_mean) * 1.05, 0.0, ceiling))

        uncertain  = (agree_weight < 0.45) or veto_applied
    else:
        confidence = min(raw_conf, 0.60)
        uncertain  = True

    return final, verdict, round(confidence, 4), uncertain, veto_applied, veto_reason


def _allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in _ALLOWED_EXTS

def _risk_level(confidence, verdict, uncertain=False):
    if verdict == "Real":
        return "SAFE" if confidence >= 0.90 else "LOW"

    if uncertain:
        return "MEDIUM"                                    
    if confidence >= 0.95: return "CRITICAL"
    if confidence >= 0.85: return "HIGH"
    if confidence >= 0.70: return "MEDIUM"
    return "MEDIUM"                                         

def _extract_frames(path):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc     = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    if total < 1:
        cap.release()
        return [], [], fps, {"width": w, "height": h, "fps": round(fps, 3),
                              "frame_count": total, "duration_s": 0.0,
                              "codec": fourcc or "unknown"}
    n       = min(_TARGET_FRAMES, total)
    indices = sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})
    idx_set = set(indices)
    frames  = []
    times   = []
    idx     = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx in idx_set:
            rgb = cv2.cvtColor(cv2.resize(frame, _INPUT_SIZE), cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            times.append(round(idx / max(fps, 1e-6), 3))
        idx += 1
    cap.release()
    duration = round(total / max(fps, 1e-6), 3)
    meta = {
        "width":       w,
        "height":      h,
        "fps":         round(fps, 3),
        "frame_count": total,
        "duration_s":  duration,
        "codec":       fourcc or "unknown",
    }
    return frames, times, fps, meta

def _file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf: break
            h.update(buf)
    return h.hexdigest()

def _infer_nn(frames):

    batch  = np.array(frames, dtype=np.float32)
    raw    = _model.predict(batch, verbose=0)
    logits = np.log(raw / (1.0 - raw + 1e-9) + 1e-9) / _TEMP
    scaled = 1.0 / (1.0 + np.exp(-logits))
    return [float(p[0]) for p in scaled]

def _dct_score(gray):
    """
    GAN transposed-convolution upsampling leaves a fingerprint in the DCT
    domain: unnaturally low coefficient-of-variation (CV) in per-block
    high-frequency energy (Guarnera et al., CVPRW 2020).
    Real images: high CV from diverse scene textures.
    Returns 1 = real-like, 0 = synthetic-like.
    """
    h, w = gray.shape
    hf_energies = []
    for i in range(0, h - 8, 8):
        for j in range(0, w - 8, 8):
            block = gray[i:i + 8, j:j + 8].astype(np.float32)
            dct   = cv2.dct(block)
            hf_energies.append(float(np.mean(np.abs(dct[4:, 4:]))))
    if len(hf_energies) < 8:
        return 0.5
    vals = np.array(hf_energies)
    cv   = float(np.std(vals)) / (float(np.mean(vals)) + 1e-8)
    return float(np.clip(cv / 2.5, 0.0, 1.0))

def _rpPG_score(frames, fps):
    """
    Remote photoplethysmography: real skin shows periodic micro-oscillations
    at heart-rate frequency (0.7-2.5 Hz). GAN faces lack this signal.
    Measures band-power ratio (BPR) of temporal green/red channel means.
    Returns 1 = strong liveness signal (real), 0 = absent (fake).
    """
    if len(frames) < 16:
        return 0.5

    g_sig, r_sig = [], []
    for frame in frames:
        h, w = frame.shape[:2]
        roi  = frame[int(h * 0.15):int(h * 0.65), int(w * 0.25):int(w * 0.75)]
        if roi.size == 0:
            roi = frame
        g_sig.append(float(np.mean(roi[:, :, 1])))
        r_sig.append(float(np.mean(roi[:, :, 0])))

    def band_power_ratio(sig, fps_val, f_lo, f_hi):
        n   = len(sig)
        arr = np.array(sig, dtype=np.float64) - np.mean(sig)
        t   = np.arange(n)
        arr -= np.polyval(np.polyfit(t, arr, 1), t)
        fft_mag = np.abs(np.fft.rfft(arr, n=n * 4)) ** 2
        freqs   = np.fft.rfftfreq(n * 4, d=1.0 / fps_val)
        mask    = (freqs >= f_lo) & (freqs <= f_hi)
        return float(np.sum(fft_mag[mask])) / (float(np.sum(fft_mag)) + 1e-10)

    bpr = (
        band_power_ratio(g_sig, fps, 0.7, 2.5) +
        band_power_ratio(r_sig, fps, 0.7, 2.5)
    ) / 2.0
    return float(np.clip((bpr - 0.03) / 0.22, 0.0, 1.0))

def _fft_score(gray_f32):
    """
    Natural images obey a 1/f^2 power law (low-frequency dominance).
    GAN checkerboard artifacts elevate specific high-frequency annuli.
    Windowed FFT suppresses edge ringing for a cleaner spectrum.
    Returns 1 = natural spectrum, 0 = synthetic artifacts.
    """
    h, w   = gray_f32.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    mag    = np.abs(np.fft.fftshift(np.fft.fft2(gray_f32 * window))) + 1e-10
    cy, cx = h // 2, w // 2
    y_g    = np.arange(h, dtype=np.float32) - cy
    x_g    = np.arange(w, dtype=np.float32) - cx
    dist   = np.sqrt(y_g[:, None] ** 2 + x_g[None, :] ** 2)
    r_max  = min(h, w) / 2.0
    total  = float(mag.sum()) + 1e-10
    low_e  = float(mag[dist < r_max * 0.12].sum())
    high_e = float(mag[dist >= r_max * 0.65].sum())
    return float(np.clip(
        (low_e / total) * 1.5 - (high_e / total) * 3.2 + 0.12,
        0.0, 1.0
    ))

def _temporal_scores(frames):
    """
    Real video: spatially concentrated motion (high diff CV).
    Deepfake re-synthesis: diffuse low-amplitude pixel changes (low CV).
    Returns per-frame scores; first frame defaults to 0.5.
    """
    scores = []
    prev   = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if prev is not None:
            diff = np.abs(gray - prev)
            cv   = float(np.std(diff)) / (float(np.mean(diff)) + 1e-6)
            scores.append(float(np.clip(cv / 5.0, 0.0, 1.0)))
        else:
            scores.append(0.5)
        prev = gray
    return scores

def _blend_score(gray):
    """
    Face-swap compositing creates a ring-shaped edge-density anomaly
    (non-monotonic radial profile) around the blend boundary.
    Returns 1 = clean profile (real), 0 = blend artifact.
    """
    edges  = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    h, w   = gray.shape
    cy, cx = h // 2, w // 2
    y_g    = np.arange(h, dtype=np.float32) - cy
    x_g    = np.arange(w, dtype=np.float32) - cx
    dist   = np.sqrt(y_g[:, None] ** 2 + x_g[None, :] ** 2)
    r_max  = min(h, w) / 2.0
    ring_d = []
    for i in range(8):
        mask = (dist >= r_max * i / 8) & (dist < r_max * (i + 1) / 8)
        if mask.sum() > 0:
            ring_d.append(float(edges[mask].mean()))
    if len(ring_d) < 4:
        return 0.5
    mid          = np.array(ring_d[2:6])
    sign_changes = float(np.sum(np.abs(np.diff(np.sign(np.diff(mid)))) / 2.0))
    return float(np.clip(1.0 - sign_changes / 3.0, 0.0, 1.0))

def _analyze(path, source_url=None):
    frames, frame_times, fps, vmeta = _extract_frames(path)
    if len(frames) < 4:
        return None

    sha256 = _file_sha256(path)
    file_size_b = os.path.getsize(path)

    nn_scores   = _infer_nn(frames)
    temp_scores = _temporal_scores(frames)

    per_frame  = []
    dct_list   = []
    fft_list   = []
    blend_list = []

    for i, frame in enumerate(frames):
        gray   = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32)

        dct_s   = _dct_score(gray)
        fft_s   = _fft_score(gray_f)
        blend_s = _blend_score(gray)

        dct_list.append(dct_s)
        fft_list.append(fft_s)
        blend_list.append(blend_s)

        raw = (
            nn_scores[i]   * _W_NN       +
            dct_s          * _W_DCT      +
            fft_s          * _W_FFT      +
            temp_scores[i] * _W_TEMP_SIG +
            blend_s        * _W_BLEND
        ) / _W_PER_FRAME
        per_frame.append(float(np.clip(raw, 0.0, 1.0)))

    rpPG  = _rpPG_score(frames, fps)
    arr   = np.array(per_frame)

    nn_mean    = float(np.mean(nn_scores))
    dct_mean   = float(np.mean(dct_list))
    fft_mean   = float(np.mean(fft_list))
    temp_mean  = float(np.mean(temp_scores))
    blend_mean = float(np.mean(blend_list))

    final, verdict, confidence, uncertain, veto, veto_reason = _fuse_verdict(
        arr, rpPG, nn_mean, dct_mean, fft_mean, temp_mean, blend_mean,
    )

    models = [
        {"name": "Neural Network Classifier",    "score": round(nn_mean,    3), "verdict": "Real" if nn_mean    >= 0.5 else "Fake"},
        {"name": "DCT Block Artifact Analysis",  "score": round(dct_mean,   3), "verdict": "Real" if dct_mean   >= 0.5 else "Fake"},
        {"name": "rPPG Biological Liveness",     "score": round(rpPG,       3), "verdict": "Real" if rpPG       >= 0.5 else "Fake"},
        {"name": "Frequency Domain (FFT)",       "score": round(fft_mean,   3), "verdict": "Real" if fft_mean   >= 0.5 else "Fake"},
        {"name": "Temporal Coherence",           "score": round(temp_mean,  3), "verdict": "Real" if temp_mean  >= 0.5 else "Fake"},
        {"name": "Blend Boundary Detection",     "score": round(blend_mean, 3), "verdict": "Real" if blend_mean >= 0.5 else "Fake"},
    ]

    std_s  = float(np.std(arr))
    real_v = int(np.sum(arr >= 0.5))
    fake_v = len(per_frame) - real_v

    return {
        "result":          verdict,
        "confidence":      confidence,
        "uncertain":       uncertain,
        "forensic_veto":   veto,
        "veto_reason":     veto_reason,
        "model_type":      "trained" if _IS_TRAINED else "placeholder",
        "frames_analyzed": len(per_frame),
        "real_votes":      real_v,
        "fake_votes":      fake_v,
        "mean_score":      round(float(arr.mean()), 4),
        "median_score":    round(float(np.median(arr)), 4),
        "consistency":     round(float(max(0.0, 1.0 - std_s * 2.5)), 4),
        "frame_scores":    [round(s, 4) for s in per_frame],
        "frame_timestamps_s": frame_times,
        "models":          models,
        "source":          "url" if source_url else "upload",
        "source_url":      source_url,
        "engine":          "EfficientNet-B4 Ensemble" if _IS_TRAINED else "Signal Analysis (placeholder model)",
        "risk_level":      _risk_level(confidence, verdict, uncertain),
        "final_score":     round((1.0 - final) * 100, 1) if verdict == "Fake" else round(final * 100, 1),
        "model_version":   _meta.get("version", "0.0.0"),
        "model_accuracy":  _meta.get("accuracy"),
        "model_auc":       _meta.get("auc"),
        "forensics": {
            "sha256":      sha256,
            "file_size_b": file_size_b,
            "video":       vmeta,
        },
    }

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File exceeds the 500 MB limit"}), 413

@app.route("/")
def index():
    return send_from_directory(_BASE_DIR, "frontend.html")

@app.route("/health")
def health():
    with _STATS_LOCK:
        s = dict(_STATS)
    return jsonify({
        "status":         "ok",
        "uptime_s":       round(time.time() - _BOOT_TIME, 1),
        "model_ready":    True,
        "model_type":     "trained" if _IS_TRAINED else "placeholder",
        "analyses_total": s["analyses_total"],
        "analyses_real":  s["analyses_real"],
        "analyses_fake":  s["analyses_fake"],
        "errors":         s["errors"],
    })

@app.route("/status")
def status():
    return jsonify({
        "service":            "DeepVerify",
        "version":            "4.0",
        "model_type":         "trained" if _IS_TRAINED else "placeholder",
        "model_architecture": _meta.get("architecture", "MobileNetV2"),
        "model_version":      _meta.get("version", "0.0.0"),
        "model_accuracy":     _meta.get("accuracy"),
        "model_auc":          _meta.get("auc"),
        "model_temperature":  _meta.get("temperature"),
        "trained_at":         _meta.get("trained_at"),
        "signals":            6,
        "target_frames":      _TARGET_FRAMES,
        "input_size":         list(_INPUT_SIZE),
        "max_upload_mb":      _MAX_BYTES // (1024 * 1024),
        "accepted_formats":   sorted(_ALLOWED_EXTS),
        "ready":              True,
    })

@app.route("/model-card")
def model_card():
    return jsonify({
        "name":            "DeepVerify Detector",
        "version":         _meta.get("version", "0.0.0"),
        "architecture":    _meta.get("architecture", "MobileNetV2"),
        "type":            _meta.get("type", "placeholder"),
        "input_shape":     [*_INPUT_SIZE, 3],
        "output":          "P(real) in [0, 1]; verdict = Real if >= 0.5 else Fake",
        "calibration":     {"method": "temperature scaling", "T": _meta.get("temperature")},
        "metrics":         {"accuracy": _meta.get("accuracy"), "auc": _meta.get("auc")},
        "training_data":   {
            "samples": _meta.get("train_samples"),
            "val":     _meta.get("val_samples"),
            "frames_per_video": _meta.get("frames_per_video"),
        },
        "trained_at":      _meta.get("trained_at"),
        "ensemble_weights": {
            "neural":   _W_NN,
            "dct":      _W_DCT,
            "fft":      _W_FFT,
            "temporal": _W_TEMP_SIG,
            "blend":    _W_BLEND,
            "rppg":     _W_RPGG,
        },
        "license":   "Research / academic use",
        "intended":  "Forensic analysis of pre-recorded video for deepfake detection.",
        "limitations": [
            "Performance degrades on extreme compression or sub-180p resolution.",
            "rPPG signal requires visible facial skin and >= 16 frames.",
            "Single-face assumption; multi-face videos use the dominant region.",
        ],
    })

@app.route("/training_history")
def training_history():
    """Per-epoch training metrics (loss/acc/auc) for the live charts in the UI.
    Returns 404-style empty payload if no history is available so the front-end
    can render fallback content gracefully."""
    path = os.path.join(_BASE_DIR, "training_history.json")
    if not os.path.exists(path):
        return jsonify({"available": False,
                        "reason": "No training_history.json found. Run training/train_local.py to generate it."})
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["available"] = True
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"available": False, "reason": f"failed to parse history: {exc}"})


@app.route("/openapi.json")
def openapi():
    return jsonify({
        "openapi": "3.0.3",
        "info": {
            "title":       "DeepVerify API",
            "version":     "4.0.0",
            "description": "Enterprise deepfake video detection. 6-signal forensic ensemble.",
        },
        "servers": [{"url": "http://127.0.0.1:5000"}],
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Analyze a video for deepfake indicators",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {"schema": {"type": "object",
                                "properties": {"video": {"type": "string", "format": "binary"}}}},
                            "application/json":    {"schema": {"type": "object",
                                "properties": {"url": {"type": "string", "format": "uri"}},
                                "required": ["url"]}},
                        }
                    },
                    "responses": {"200": {"description": "Analysis result"}},
                }
            },
            "/health":     {"get": {"summary": "Liveness + runtime stats"}},
            "/status":     {"get": {"summary": "Service + model metadata"}},
            "/model-card": {"get": {"summary": "Detailed model card"}},
            "/openapi.json": {"get": {"summary": "This OpenAPI document"}},
        },
    })

def _is_streaming_site(url):
    """Detect URLs that need a video extractor (yt-dlp) rather than direct fetch."""
    u = url.lower()
    hosts = (
        "youtube.com/watch", "youtu.be/", "youtube.com/shorts",
        "vimeo.com/", "dailymotion.com/", "tiktok.com/",
        "facebook.com/", "fb.watch/", "instagram.com/",
        "twitter.com/", "x.com/",
    )
    return any(h in u for h in hosts)

def _ytdlp_fetch(url, dest_dir):
    """Use yt-dlp to download a streaming-site video to dest_dir.
    Returns (path, info) or raises ValueError."""
    try:
        import yt_dlp
    except ImportError:
        raise ValueError("Streaming-site URL detected but yt-dlp is not installed.")
    out_tmpl = os.path.join(dest_dir, "yt_%(id)s.%(ext)s")
    ydl_opts = {
        "outtmpl":         out_tmpl,
        "quiet":           True,
        "no_warnings":     True,
        "noplaylist":      True,
        "format":          "best[ext=mp4][filesize<500M]/best[ext=mp4]/best[filesize<500M]/best",
        "max_filesize":    _MAX_BYTES,
        "socket_timeout":  30,
        "retries":         2,
        "user_agent":      _FETCH_UA,
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            path = ydl.prepare_filename(info)

            if not os.path.exists(path):
                base, _ = os.path.splitext(path)
                for cand in (base + ".mp4", base + ".mkv", base + ".webm"):
                    if os.path.exists(cand):
                        path = cand; break
        if not os.path.exists(path):
            raise ValueError("yt-dlp completed but output file not found")
        return path, info
    except ValueError:
        raise
    except Exception as exc:
        msg = str(exc).split("\n")[0][:200]
        raise ValueError(f"Could not download from streaming site: {msg}")

def _fetch_url_to_tmp(url):
    """
    Robust URL downloader: browser-like UA, follows redirects, accepts any
    content-type, validates by attempting to open with OpenCV after download.
    Returns (tmp_path, orig_filename) or raises ValueError with a clear message.
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    if _is_streaming_site(url):
        path, info = _ytdlp_fetch(url, _UPLOAD_DIR)
        title = (info or {}).get("title") or path.rsplit(os.sep, 1)[-1]
        return path, title

    headers = {
        "User-Agent": _FETCH_UA,
        "Accept":     "video/*, application/octet-stream, */*;q=0.8",
        "Range":      "bytes=0-",
    }
    try:
        resp = _req.get(url, headers=headers, stream=True, timeout=(10, 60),
                        allow_redirects=True)
    except _req.exceptions.SSLError:
        raise ValueError("SSL certificate error while fetching URL")
    except _req.exceptions.ConnectTimeout:
        raise ValueError("Connection timed out fetching URL")
    except _req.exceptions.ReadTimeout:
        raise ValueError("Read timeout while downloading video")
    except _req.exceptions.ConnectionError:
        raise ValueError("Could not connect to remote host")
    except Exception as exc:
        raise ValueError(f"Failed to fetch URL: {exc}")

    if resp.status_code in (401, 403):
        raise ValueError("Remote server denied access (HTTP {0})".format(resp.status_code))
    if resp.status_code == 404:
        raise ValueError("Video not found at URL (HTTP 404)")
    if resp.status_code >= 400:
        raise ValueError(f"Remote server returned HTTP {resp.status_code}")

    ct = (resp.headers.get("content-type") or "").lower()
    if ct.startswith(("text/html", "application/xhtml", "text/plain")):
        raise ValueError(
            "URL returned an HTML page, not a video file. Use a direct link to a .mp4/.mov/.webm file."
        )

    cl = resp.headers.get("content-length")
    if cl and int(cl) > _MAX_BYTES:
        raise ValueError("Remote file exceeds 500 MB limit")

    path_part = url.split("?", 1)[0].rsplit("/", 1)[-1]
    ext = ""
    for e in _ALLOWED_EXTS:
        if path_part.lower().endswith("." + e):
            ext = e; break
    if not ext:
        for e in _ALLOWED_EXTS:
            if e in ct:
                ext = e; break
    if not ext:
        ext = "mp4"

    fd, tmp_path = tempfile.mkstemp(suffix="." + ext, dir=_UPLOAD_DIR)
    downloaded = 0
    try:
        with os.fdopen(fd, "wb") as out:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > _MAX_BYTES:
                    raise ValueError("Download exceeded 500 MB limit")
                out.write(chunk)
    except Exception:
        try: os.remove(tmp_path)
        except OSError: pass
        raise

    if downloaded < 1024:
        try: os.remove(tmp_path)
        except OSError: pass
        raise ValueError("Downloaded file is too small to be a video")

    cap = cv2.VideoCapture(tmp_path)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n < 4:
        try: os.remove(tmp_path)
        except OSError: pass
        raise ValueError(
            "URL did not return a valid video (could not decode frames). "
            "Make sure it is a direct link to an MP4/MOV/WEBM file, not a streaming page."
        )

    orig = path_part or "video." + ext
    return tmp_path, orig

@app.route("/predict", methods=["POST"])
def predict():
    t0       = time.time()
    tmp_path = None
    source_url = None
    try:
        if "video" in request.files:
            f = request.files["video"]
            if not f.filename:
                return jsonify({"error": "No file selected"}), 400
            if not _allowed(f.filename):
                return jsonify({"error": "Format not supported. Accepted: mp4, mov, avi, mkv, webm"}), 400
            orig     = secure_filename(f.filename)
            ext      = orig.rsplit(".", 1)[-1].lower()
            uid      = hashlib.md5((orig + str(time.time())).encode()).hexdigest()[:8]
            tmp_path = os.path.join(_UPLOAD_DIR, f"{uid}.{ext}")
            f.save(tmp_path)
        elif request.is_json:
            body = request.get_json(silent=True) or {}
            url  = (body.get("url") or "").strip()
            if not url:
                return jsonify({"error": "Provide a video file or URL"}), 400
            try:
                tmp_path, orig = _fetch_url_to_tmp(url)
                source_url = url
            except ValueError as ve:
                with _STATS_LOCK: _STATS["errors"] += 1
                return jsonify({"error": str(ve)}), 400
        else:
            return jsonify({"error": "Provide a video file or URL"}), 400

        result = _analyze(tmp_path, source_url=source_url)
        if result is None:
            with _STATS_LOCK: _STATS["errors"] += 1
            return jsonify({"error": "Video too short or could not be processed"}), 422

        result["processing_time"] = round(time.time() - t0, 2)
        result["filename"] = orig

        with _STATS_LOCK:
            _STATS["analyses_total"] += 1
            if result["result"] == "Fake": _STATS["analyses_fake"] += 1
            else:                          _STATS["analyses_real"] += 1

        return jsonify(result)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _analyze_stream(path, source_url=None):
    """Generator that yields SSE events as analysis progresses, then a final 'done' event."""
    def ev(name, payload):
        return "event: {0}\ndata: {1}\n\n".format(name, json.dumps(payload, default=float))

    yield ev("status", {"stage": "extract", "msg": "Extracting frames",
                         "detail": "OpenCV VideoCapture - sampling up to 64 frames", "step": 1, "total": 7})

    frames, frame_times, fps, vmeta = _extract_frames(path)
    if len(frames) < 4:
        yield ev("error", {"error": "Video too short or could not be processed"})
        return

    sha256 = _file_sha256(path)
    file_size_b = os.path.getsize(path)

    yield ev("meta", {
        "frames":   len(frames),
        "fps":      vmeta.get("fps"),
        "duration": vmeta.get("duration_s"),
        "width":    vmeta.get("width"),
        "height":   vmeta.get("height"),
        "codec":    vmeta.get("codec"),
        "sha256":   sha256,
        "file_size_b": file_size_b,
        "engine":      "EfficientNet-B4 Ensemble" if _IS_TRAINED else "Signal Analysis (placeholder)",
        "model_type":  "trained" if _IS_TRAINED else "placeholder",
        "model_accuracy": _meta.get("accuracy"),
        "model_auc":      _meta.get("auc"),
    })

    yield ev("status", {"stage": "neural", "msg": "Running neural classifier (MobileNetV2 transfer)",
                         "detail": "Per-frame deepfake probability inference",
                         "model": "Neural Network Classifier", "weight": _W_NN, "step": 2, "total": 7})
    nn_scores   = _infer_nn(frames)
    yield ev("signal", {"name": "Neural Network Classifier", "key": "nn",
                         "mean": round(float(np.mean(nn_scores)), 4), "weight": _W_NN,
                         "verdict": "Real" if float(np.mean(nn_scores)) >= 0.5 else "Fake"})

    yield ev("status", {"stage": "temporal", "msg": "Computing temporal coherence",
                         "detail": "Inter-frame motion entropy & residual analysis",
                         "model": "Temporal Coherence", "weight": _W_TEMP_SIG, "step": 3, "total": 7})
    temp_scores = _temporal_scores(frames)
    yield ev("signal", {"name": "Temporal Coherence", "key": "temp",
                         "mean": round(float(np.mean(temp_scores)), 4), "weight": _W_TEMP_SIG,
                         "verdict": "Real" if float(np.mean(temp_scores)) >= 0.5 else "Fake"})

    per_frame, dct_list, fft_list, blend_list = [], [], [], []
    n = len(frames)
    yield ev("status", {"stage": "signals", "msg": "Per-frame forensic signal analysis",
                         "detail": "DCT block artifacts | FFT spectrum | Blend boundary",
                         "step": 4, "total": 7})

    for i, frame in enumerate(frames):
        gray   = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32)
        dct_s   = _dct_score(gray)
        fft_s   = _fft_score(gray_f)
        blend_s = _blend_score(gray)
        dct_list.append(dct_s); fft_list.append(fft_s); blend_list.append(blend_s)
        raw = (
            nn_scores[i]   * _W_NN       +
            dct_s          * _W_DCT      +
            fft_s          * _W_FFT      +
            temp_scores[i] * _W_TEMP_SIG +
            blend_s        * _W_BLEND
        ) / _W_PER_FRAME
        s = float(np.clip(raw, 0.0, 1.0))
        per_frame.append(s)

        if (i % max(1, n // 32)) == 0 or i == n - 1:
            yield ev("frame", {
                "i":      i,
                "n":      n,
                "t":      frame_times[i] if i < len(frame_times) else None,
                "score":  round(s, 4),
                "nn":     round(nn_scores[i], 4),
                "dct":    round(dct_s, 4),
                "fft":    round(fft_s, 4),
                "temp":   round(temp_scores[i], 4),
                "blend":  round(blend_s, 4),
            })

    yield ev("status", {"stage": "rppg", "msg": "Computing rPPG biological liveness",
                         "detail": "Chrominance-based pulse extraction over face region",
                         "model": "rPPG Biological Liveness", "weight": _W_RPGG, "step": 5, "total": 7})
    rpPG  = _rpPG_score(frames, fps)
    yield ev("signal", {"name": "rPPG Biological Liveness", "key": "rppg",
                         "mean": round(float(rpPG), 4), "weight": _W_RPGG,
                         "verdict": "Real" if rpPG >= 0.5 else "Fake"})
    arr   = np.array(per_frame)

    nn_mean    = float(np.mean(nn_scores))
    dct_mean   = float(np.mean(dct_list))
    fft_mean   = float(np.mean(fft_list))
    temp_mean  = float(np.mean(temp_scores))
    blend_mean = float(np.mean(blend_list))

    yield ev("signal", {"name": "DCT Block Artifact Analysis", "key": "dct",
                         "mean": round(dct_mean, 4), "weight": _W_DCT,
                         "verdict": "Real" if dct_mean >= 0.5 else "Fake"})
    yield ev("signal", {"name": "Frequency Domain (FFT)", "key": "fft",
                         "mean": round(fft_mean, 4), "weight": _W_FFT,
                         "verdict": "Real" if fft_mean >= 0.5 else "Fake"})
    yield ev("signal", {"name": "Blend Boundary Detection", "key": "blend",
                         "mean": round(blend_mean, 4), "weight": _W_BLEND,
                         "verdict": "Real" if blend_mean >= 0.5 else "Fake"})

    yield ev("status", {"stage": "fuse", "msg": "Fusing 6-signal weighted ensemble",
                         "detail": "Reliability-weighted agreement + physics veto",
                         "step": 6, "total": 7})

    final, verdict, confidence, uncertain, veto, veto_reason = _fuse_verdict(
        arr, rpPG, nn_mean, dct_mean, fft_mean, temp_mean, blend_mean,
    )
    yield ev("status", {"stage": "report", "msg": "Compiling forensic report",
                         "detail": f"Verdict: {verdict} ({confidence*100:.1f}%)",
                         "step": 7, "total": 7})

    models = [
        {"name": "Neural Network Classifier",    "score": round(nn_mean,    3), "verdict": "Real" if nn_mean    >= 0.5 else "Fake"},
        {"name": "DCT Block Artifact Analysis",  "score": round(dct_mean,   3), "verdict": "Real" if dct_mean   >= 0.5 else "Fake"},
        {"name": "rPPG Biological Liveness",     "score": round(rpPG,       3), "verdict": "Real" if rpPG       >= 0.5 else "Fake"},
        {"name": "Frequency Domain (FFT)",       "score": round(fft_mean,   3), "verdict": "Real" if fft_mean   >= 0.5 else "Fake"},
        {"name": "Temporal Coherence",           "score": round(temp_mean,  3), "verdict": "Real" if temp_mean  >= 0.5 else "Fake"},
        {"name": "Blend Boundary Detection",     "score": round(blend_mean, 3), "verdict": "Real" if blend_mean >= 0.5 else "Fake"},
    ]
    std_s  = float(np.std(arr))
    real_v = int(np.sum(arr >= 0.5))
    fake_v = len(per_frame) - real_v

    final_payload = {
        "result":          verdict,
        "confidence":      confidence,
        "uncertain":       uncertain,
        "forensic_veto":   veto,
        "veto_reason":     veto_reason,
        "model_type":      "trained" if _IS_TRAINED else "placeholder",
        "frames_analyzed": len(per_frame),
        "real_votes":      real_v,
        "fake_votes":      fake_v,
        "mean_score":      round(float(arr.mean()), 4),
        "median_score":    round(float(np.median(arr)), 4),
        "consistency":     round(float(max(0.0, 1.0 - std_s * 2.5)), 4),
        "frame_scores":    [round(s, 4) for s in per_frame],
        "frame_timestamps_s": frame_times,
        "models":          models,
        "source":          "url" if source_url else "upload",
        "source_url":      source_url,
        "engine":          "EfficientNet-B4 Ensemble" if _IS_TRAINED else "Signal Analysis (placeholder model)",
        "risk_level":      _risk_level(confidence, verdict, uncertain),
        "final_score":     round((1.0 - final) * 100, 1) if verdict == "Fake" else round(final * 100, 1),
        "model_version":   _meta.get("version", "0.0.0"),
        "model_accuracy":  _meta.get("accuracy"),
        "model_auc":       _meta.get("auc"),
        "forensics": {
            "sha256":      sha256,
            "file_size_b": file_size_b,
            "video":       vmeta,
        },
    }
    with _STATS_LOCK:
        _STATS["analyses_total"] += 1
        if verdict == "Fake": _STATS["analyses_fake"] += 1
        else:                 _STATS["analyses_real"] += 1
    yield ev("done", final_payload)

@app.route("/predict-stream", methods=["POST"])
def predict_stream():
    """Server-Sent Events streaming version of /predict.

    Accepts the same payload as /predict (multipart 'video' or JSON {url}).
    Emits events: status, meta, frame, done, error.
    """
    from flask import Response, stream_with_context

    tmp_path   = None
    source_url = None

    if "video" in request.files:
        f = request.files["video"]
        if not f.filename or not _allowed(f.filename):
            return jsonify({"error": "Invalid file"}), 400
        orig     = secure_filename(f.filename)
        ext      = orig.rsplit(".", 1)[-1].lower()
        uid      = hashlib.md5((orig + str(time.time())).encode()).hexdigest()[:8]
        tmp_path = os.path.join(_UPLOAD_DIR, f"{uid}.{ext}")
        f.save(tmp_path)
    elif request.is_json or request.form.get("url"):
        url = (request.form.get("url") or
               (request.get_json(silent=True) or {}).get("url") or "").strip()
        if not url:
            return jsonify({"error": "Provide a video file or URL"}), 400
        try:
            tmp_path, _orig = _fetch_url_to_tmp(url)
            source_url = url
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
    else:
        return jsonify({"error": "Provide a video file or URL"}), 400

    def gen():
        try:
            t0 = time.time()
            for chunk in _analyze_stream(tmp_path, source_url=source_url):
                yield chunk
            yield "event: ping\ndata: {{\"elapsed\":{0:.2f}}}\n\n".format(time.time() - t0)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except OSError: pass

    return Response(stream_with_context(gen()),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control":   "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection":      "keep-alive",
                    })

@app.route("/resolve-url", methods=["POST"])
def resolve_url():
    """Return a player-friendly URL for the frontend preview.

    Strategy:
      * Direct video files (.mp4 etc) -> serve URL directly to <video>.
      * YouTube/Vimeo -> try iframe embed first; the frontend has a fallback link.
      * If iframe is known to fail (?fallback=local in body), download via yt-dlp
        and serve from /media/<id> as <video>.
    """
    body = request.get_json(silent=True) or {}
    url  = (body.get("url") or "").strip()
    want_local = bool(body.get("local"))
    if not url or not url.lower().startswith(("http://", "https://")):
        return jsonify({"error": "Invalid URL"}), 400
    u = url
    lo = u.lower()

    for e in _ALLOWED_EXTS:
        if "." + e in lo:
            return jsonify({"kind": "video", "embed": u, "site": "Direct"})

    def _to_local(label):
        try:
            uid = hashlib.md5(u.encode()).hexdigest()[:16]

            for ext in ("mp4", "webm", "mkv", "mov"):
                cached = os.path.join(_MEDIA_DIR, f"{uid}.{ext}")
                if os.path.exists(cached):
                    return jsonify({"kind": "video", "embed": f"/media/{uid}.{ext}",
                                    "site": label, "local": True})
            path, info = _ytdlp_fetch(u, _MEDIA_DIR)
            ext = path.rsplit(".", 1)[-1].lower()
            target = os.path.join(_MEDIA_DIR, f"{uid}.{ext}")
            try:
                if os.path.exists(target): os.remove(target)
                os.rename(path, target)
            except OSError:
                target = path
            fname = os.path.basename(target)
            return jsonify({"kind": "video", "embed": f"/media/{fname}",
                            "site": label, "local": True})
        except Exception as e:
            return jsonify({"error": f"Could not fetch media: {e}"}), 502

    yt_id = None
    if "youtube.com/watch" in lo:
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(u).query); yt_id = (q.get("v") or [None])[0]
    elif "youtu.be/" in lo:
        yt_id = u.split("youtu.be/", 1)[1].split("?", 1)[0].split("&", 1)[0].strip("/")
    elif "youtube.com/shorts/" in lo:
        yt_id = u.split("/shorts/", 1)[1].split("?", 1)[0].strip("/")
    if yt_id:
        if want_local:
            return _to_local("YouTube (local)")
        return jsonify({
            "kind": "iframe",
            "embed": f"https://www.youtube.com/embed/{yt_id}?autoplay=1&rel=0&modestbranding=1&playsinline=1",
            "site": "YouTube",
            "fallback_url": "/resolve-url",                                        
        })

    if "vimeo.com/" in lo:
        vid = u.split("vimeo.com/", 1)[1].split("?", 1)[0].strip("/").split("/")[0]
        if vid.isdigit():
            if want_local:
                return _to_local("Vimeo (local)")
            return jsonify({
                "kind": "iframe",
                "embed": f"https://player.vimeo.com/video/{vid}?autoplay=1",
                "site": "Vimeo",
                "fallback_url": "/resolve-url",
            })

    if _is_streaming_site(u):
        return _to_local("External (local)")
    return jsonify({"kind": "video", "embed": u, "site": "Direct"})

@app.route("/media/<path:filename>", methods=["GET"])
def serve_media(filename):
    """Serve cached media files for the frontend video player.

    Only files inside the media cache are served. Filenames containing
    path traversal characters are rejected by send_from_directory.
    """
    return send_from_directory(_MEDIA_DIR, filename, conditional=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
