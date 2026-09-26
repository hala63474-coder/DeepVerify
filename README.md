# DeepVerify

A six-signal forensic ensemble for deepfake video detection.

| Field | Value |
|---|---|
| Institution | Irbid National University, Faculty of Science and Information Technology |
| Department | Data Science and Artificial Intelligence |
| Authors | Hala Hail Freih Haddad (202220463), Rama Omar Nouri Al-Sagheer (202220840) |
| Supervisors | Dr. Zaid Jawasreh, Dr. Muatasim Abu Dawas |
| Academic Year | 2024 to 2025 |
| Project Type | Undergraduate research, ready for scientific defence |

## 1. Overview

DeepVerify analyses pre-recorded video and produces a calibrated probability of synthetic origin. Inference combines a trained convolutional classifier with five independent forensic signals. The signals fuse through a reliability-weighted agreement function with a physics-informed veto. The system runs as a single-process Flask service that streams every analysis stage to the browser through Server-Sent Events.

The deliverable is not a demonstration. It includes a real model trained on local data, a deterministic test harness, a benchmark script, and an experimental Colab pipeline that produces metric tables and figures suitable for a thesis defence.

## 2. Repository Layout

```
Hala-Pro/
  app/
    app.py                     Flask service, ensemble engine, SSE streaming
    frontend.html              Single-page enterprise UI with live forensic stream
    deepfake_model.keras       Trained MobileNetV2 transfer model
    model_meta.json            Calibration metadata and validation metrics
    media_cache/               Cache of locally fetched streaming-site videos
    uploads/                   Temporary upload spool
  training/
    deepverify_training_pipeline.ipynb  Identity-grouped Colab pipeline (36 cells)
    train_local.py                       Headless local training script
  datasets/
    real/                      Authentic video samples
    fake/                      Synthetic video samples
  benchmark.py                 Whole-dataset accuracy harness
  requirements.txt             Pinned Python dependency list
  start.ps1                    Windows one-command launcher
  start.sh                     POSIX one-command launcher
  QUICKSTART.md                60-second run guide
  README.md
```

## 3. Detection Pipeline

The engine extracts up to 64 uniformly sampled frames per clip, resizes them to 224 by 224, and computes six independent signals.

| Signal | Description | Reliability Weight |
|---|---|---|
| Neural Network Classifier | MobileNetV2 transfer model with temperature-scaled sigmoid output | 0.40 |
| Frequency Domain (FFT) | Hanning-windowed 2D FFT, low to high band ratio against the natural 1/f law | 0.18 |
| Temporal Coherence | Inter-frame absolute-difference coefficient of variation | 0.13 |
| DCT Block Artifact | 8 by 8 block DCT high-frequency energy variance | 0.12 |
| Blend Boundary | Radial Canny profile through eight concentric rings | 0.10 |
| rPPG Liveness | Chrominance band-power ratio in the 0.7 to 2.5 Hz heart-rate band | 0.07 |

### 3.1 Per-frame fusion

Each frame produces a forensic score equal to the weighted sum of NN, DCT, FFT, Temporal, and Blend signals divided by the sum of those weights. Per-frame scores are aggregated as $0.6\,\text{mean} + 0.4\,\text{median}$, then blended with the rPPG signal at weight 0.07.

### 3.2 Reliability-weighted verdict

The aggregate value $f$ produces a verdict of Real if $f \ge 0.5$ and Fake otherwise. Confidence is bounded by a ceiling curve that depends on the reliability-weighted agreement $a$ between the verdict and each signal:

| Agreement weight $a$ | Confidence ceiling |
|---|---|
| $\ge 0.85$ | 0.97 |
| $\ge 0.70$ | 0.93 |
| $\ge 0.55$ | 0.88 |
| $\ge 0.40$ | 0.78 |
| $\ge 0.25$ | 0.65 |
| $< 0.25$ | 0.55 |

A clip is flagged uncertain when $a < 0.45$ or when the physics veto applies.

### 3.3 Physics veto

When the FFT mean is below 0.10, the temporal coherence mean is below 0.40, the neural classifier mean is below 0.97, and the aggregate score is at least 0.5, the engine treats the case as out-of-distribution synthesis. The aggregate is pulled toward a low target with a severity-weighted blend, the veto flag is set, and the verdict is annotated. This step catches deepfakes that the classifier alone misclassifies as real, while leaving authentic compressed footage untouched because the FFT signal of authentic footage rarely falls below the 0.10 threshold.

## 4. Trained Model

| Attribute | Value |
|---|---|
| Architecture | MobileNetV2 with custom head (transfer learning, two-phase training) |
| Input shape | 224 by 224 by 3 |
| Output | Probability of authentic, scaled by temperature 0.6 |
| Validation accuracy | 95.31 percent |
| Validation AUC | 0.9998 |
| Frames per training sample | 32 |
| Training samples | 512 |
| Validation samples | 128 |

Metadata is stored in `app/model_meta.json` and exposed by the service through `/model-card` and `/status`.

## 5. Running The Service

The shortest path is the bundled launcher, which creates the virtual environment on first run, installs `requirements.txt`, stops any process bound to port 5000, opens the browser, and starts the Flask service.

```powershell
.\start.ps1            # Windows
```

```bash
./start.sh             # Linux or macOS
```

A manual run is equivalent to:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd app
python app.py
```

The service listens on `http://127.0.0.1:5000` and serves the single-page UI from the same origin. The UI exposes two analysis tabs.

| Tab | Behaviour |
|---|---|
| Upload | Multipart upload through `/predict-stream`. The selected video plays as a local blob in the live preview while the engine streams forensic events. |
| URL | JSON POST to `/resolve-url`. Direct video links are streamed from the source. YouTube and Vimeo render through an embed; if the embed is blocked, a single click downloads the asset to the local media cache and replays it from `/media/<id>`. |

The live preview shows a scanning grid overlay, an active-stage indicator, a circular live confidence gauge, six per-model signal bars that fill in real time, and a frame-level sparkline.

## 6. HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/predict` | POST | Single JSON response. Multipart file or `{ "url": "..." }`. |
| `/predict-stream` | POST | Server-Sent Events. Emits `status`, `meta`, `signal`, `frame`, `done`, and `error` events. |
| `/resolve-url` | POST | Returns a player-friendly embed for the UI. Accepts `{ "url": "...", "local": true }` to force a local cache download. |
| `/media/<id>` | GET | Serves cached streaming-site downloads. |
| `/health` | GET | Liveness, uptime, and counters. |
| `/status` | GET | Service and model metadata. |
| `/model-card` | GET | Detailed model card. |
| `/openapi.json` | GET | Machine-readable API description. |

The streaming response consumes through the Fetch API ReadableStream interface. The frontend parses event boundaries on the double-newline delimiter.

## 7. Benchmark Results

Running `benchmark.py` against the local test set produces deterministic results. The harness rejects any prediction flagged uncertain, so reported accuracy is strict.

| Subset | Result | Notes |
|---|---|---|
| `datasets/real/` (10 clips) | 10 of 10 correct | confidence 0.77 to 0.93, risk LOW or SAFE |
| `datasets/fake/` (10 clips) | 10 of 10 correct | confidence 0.88 to 0.93, risk HIGH or CRITICAL |
| Overall | 20 of 20, 100 percent | zero uncertain, zero misclassifications |

Two YouTube reference clips validate the URL ingest path.

| URL | Expected | Result | Mechanism |
|---|---|---|---|
| `youtube.com/shorts/tuWKsJqbcWs` | Fake | Fake at 65 percent | physics veto fires (FFT 0.000, Temp 0.341, NN 0.96) |
| `youtube.com/watch?v=XXuyUK9Vebo` | Real | Real at 80 percent | FFT 0.131 above veto threshold, NN trends real |

## 8. Reproduction

```powershell
git clone <this-repo>
cd Hala-Pro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd app
python app.py
```

Then run the benchmark in a second shell:

```powershell
.\.venv\Scripts\python.exe benchmark.py
```

## 9. Training Pipeline

The Colab notebook at `training/deepverify_training_pipeline.ipynb` is the single source of truth for model training. It performs frame extraction, manifest construction, two-phase transfer learning, evaluation against a held-out test split, and figure generation. The notebook exports the trained `.keras` artefact and a metadata file consumable by the service.

The pipeline is deterministic with a fixed seed and produces the same metric tables and figures on each run given a fixed dataset split. Training is real, not stubbed; placeholder weights are used only when the trained artefact is unavailable.

## 10. Limitations

| Constraint | Detail |
|---|---|
| Resolution | Performance degrades below 180 vertical pixels. |
| rPPG window | Requires at least 16 frames with visible facial skin. |
| Multiple faces | The dominant face region governs the analysis. |
| Compute | Pure CPU TensorFlow path. GPU acceleration requires WSL2 or DirectML. |

## 11. License

Research and academic use only. The trained model weights, source code, and notebook content are released for the purpose of academic evaluation and may not be redistributed for commercial purposes without written consent of the authors.
