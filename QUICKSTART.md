# DeepVerify Quick Start

## 1. Run the application (one command)

### Windows (PowerShell)
```powershell
.\start.ps1
```

### Linux / macOS
```bash
chmod +x start.sh
./start.sh
```

The script creates the virtual environment on first run, installs dependencies from
[requirements.txt](requirements.txt), kills any process already bound to port 5000,
launches the Flask service, and opens the UI in the default browser at
`http://127.0.0.1:5000/`.

## 2. Manual run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd app
python app.py
```

Open `http://127.0.0.1:5000/` in any modern browser.

## 3. Verify the service

```powershell
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/status
```

`model_type` should report `trained` and `model_ready` should be `true`.

## 4. Run the local benchmark

```powershell
.\.venv\Scripts\python.exe benchmark.py
```

Expected output on the bundled twenty-clip evaluation set: `20/20 = 100.0%`.

## 5. Train your own model (Google Colab)

Open [training/deepverify_training_pipeline.ipynb](training/deepverify_training_pipeline.ipynb)
in Google Colab. The notebook performs identity-grouped splitting, two-phase transfer
learning, and produces a `.keras` artefact plus a `model_meta.json` calibration file.
Drop both files into `app/` and restart the service to deploy.

## 6. Project layout

| Path | Purpose |
|---|---|
| `app/app.py` | Flask service, six-signal ensemble, SSE stream |
| `app/frontend.html` | Single-page UI |
| `app/deepfake_model.keras` | Trained model artefact |
| `app/model_meta.json` | Calibration and metric metadata |
| `training/deepverify_training_pipeline.ipynb` | Reproducible Colab pipeline |
| `training/train_local.py` | Headless training script |
| `benchmark.py` | Whole-folder evaluation harness |
| `requirements.txt` | Pinned dependency list |
| `start.ps1` / `start.sh` | One-command launchers |

## 7. Troubleshooting

| Symptom | Resolution |
|---|---|
| `ModuleNotFoundError: tensorflow` | Run `pip install -r requirements.txt` inside the active venv. |
| Port 5000 already in use | The launcher stops the previous process automatically; otherwise kill it manually with `Stop-Process -Id (Get-NetTCPConnection -LocalPort 5000 -State Listen).OwningProcess -Force`. |
| `model_type: placeholder` in `/health` | The trained `.keras` file is missing from `app/`. Re-train via the Colab notebook or copy it back from the release archive. |
| Browser shows the page but URL or upload returns nothing | Hard refresh with `Ctrl+Shift+R` to reload the cached JavaScript. |
