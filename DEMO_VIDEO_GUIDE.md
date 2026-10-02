# Video demo guide (target: 6–7 minutes)

Record your own screen and voice, showing the running project. The video is **not included** because a live narrated recording must be made by the presenter. Keep the recording between 5 and 10 minutes.

| Time | Show on screen | Say or demonstrate |
|---|---|---|
| 0:00–0:40 | Repository files and README | State that this is the ML-track prototype and all data are synthetic. |
| 0:40–1:40 | Notebook: data generation and charts | Show RIR distribution, feature relationship, and causal four-repetition windows. |
| 1:40–2:45 | Notebook: model architecture and split | Explain Dense 32/16 ReLU, linear output, Adam, participant-held-out test. |
| 2:45–3:35 | Notebook: metrics | Show the MAE comparison and warning precision/recall. State that 7.7% does not reach the 15% target. |
| 3:35–5:20 | Running web interface | Click Analyze next rep several times, change model, show warning, reset. Explain that values are generated features. |
| 5:20–6:10 | API reference and `/api/metrics` | Show the JSON endpoints and one prediction request if comfortable. |
| 6:10–6:50 | README deployment plan | Describe replacing synthetic features with real consented video and measuring latency. |

Suggested opening: “This is RIR Lens, a prototype for estimating repetitions in reserve. I’ll demonstrate the notebook, held-out synthetic evaluation, interface, and API. Real camera data collection is future work.”

Suggested command sequence before recording:

```bash
pip install -r requirements.txt
python server.py
```

Then open <http://127.0.0.1:8000> and `ModelNotebook.ipynb` in JupyterLab. Increase browser zoom enough for the metrics to remain legible in the recording.
