# RIR Lens — ML track prototype

**Predicting repetitions in reserve during back squats and bench presses.** This is a demonstrable concept based on Nick-Lemy Kayiranga Niyigena's September 2026 proposal. It uses **synthetic generated features only**. It does not process camera video, estimate physiological tremor, or establish real-world accuracy.

## What is included

- `ModelNotebook.ipynb`: data distributions, feature relationships, causal feature engineering, model architecture, training and held-out metrics.
- `src/pipeline.py`: synthetic data generator, four-repetition windows, participant split, NumPy neural network, evaluation, inference.
- `train.py`: reproducibly rebuilds the dataset, model files, metrics and SVG charts.
- `server.py` and `web/`: local web demo and JSON API.
- `assets/`: charts and interface design screenshot.
- `DEMO_VIDEO_GUIDE.md`: a 5–10 minute screen-recording plan.

## GitHub repository

<https://github.com/Nick-Lemy/rir-lens-ml-prototype>

## Set up and run

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python server.py
```

Open <http://127.0.0.1:8000>. For the notebook, run `jupyter lab ModelNotebook.ipynb`. On Windows use `.venv\Scripts\activate` for activation.

The checked-in generated files let the web demo start after dependency installation without retraining. `python train.py` reproduces them with the fixed generator and training seeds.

## Interface and API

The web interface steps through one synthetic squat set from a held-out simulated participant. Select the combined or velocity-only model, analyze each completed repetition, and inspect the numerical estimate, velocity, velocity loss, candidate tremor feature and RIR≤2 warning. Select **API reference** in the interface for endpoint documentation. `GET /api/demo` supplies a demo set; `GET /api/metrics` supplies the synthetic evaluation; `POST /api/predict` accepts 1–4 completed repetition records.

## Model and initial metrics

Both models use a causal history of up to four completed repetitions, plus exercise, load fraction and current repetition number. One uses velocity and velocity loss. The other adds synthetic candidate tremor amplitude and band power. Each model is a small multilayer perceptron: flattened history and mask → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, linear). Training uses mean squared error, Adam and validation early stopping. This is a compact architecture for the prototype; the proposal's TCN comparison remains future work.

24 simulated participants are split by participant ID: 16 train, 4 validation, 4 test. All normalization is fitted on training participants only. The held-out synthetic test contains 354 repetitions. See `metrics.json` for the current exact results. The primary metric is MAE in repetitions; warning precision, recall and F1 use estimated RIR≤2. The 15% improvement target in the proposal was **not** met in this synthetic run. Synthetic performance must not be used as evidence of clinical, training or video-system performance.

## Designs

- `assets/interface.png` — screenshot of the local web UI.
- `assets/rir_distribution.svg` — label distribution.
- `assets/feature_means.svg` — example feature summary.
- `assets/model_mae.svg` — held-out MAE comparison.

## Deployment plan

1. For this submission, serve the local API and static web interface with `python server.py`; screen-record the walkthrough.
2. For a hosted MVP, move the same `/api/predict` contract to FastAPI behind HTTPS and host the static frontend separately. Add authentication and per-session data controls before handling recordings.
3. After ethical approval and collection, replace generated features with consented smartphone video processing (plate tracking, pose landmarks, quality checks), retrain with participant-grouped validation, and measure end-to-display latency on the target phone and server.
4. Evaluate the proposed velocity-only versus velocity-plus-candidate-tremor TCNs on real participants. Report unavailable predictions and test whether the 15% MAE improvement and 2-second p95 latency targets are met.

No public deployment is claimed. The current server binds to `127.0.0.1` for a local demonstration.

## Limitations

The source proposal describes a future supervised data collection study; it does not provide a usable training dataset. This package therefore does not claim camera analysis, real-time latency, safety advice or participant study results. The demo's RIR labels come from the synthetic set endpoint (`total repetitions − current repetition`), but final set length is never an input to the predictor.
