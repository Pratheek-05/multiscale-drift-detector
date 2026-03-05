# 🌊 Multiscale Drift Detection Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

**A hybrid concept drift detection system combining deep learning, statistical methods, and reinforcement learning — with a live Streamlit dashboard.**

[Live Demo](https://hybrid-drift-detection.onrender.com) · [Report Bug](https://github.com/Pratheek-05/multiscale-drift-detector/issues) · [GitHub Actions](https://github.com/Pratheek-05/multiscale-drift-detector/actions)

</div>

---

##  What is Concept Drift?

In machine learning, **concept drift** happens when the statistical properties of the data a model was trained on change over time — causing model performance to degrade. This project detects drift in real time as data streams in batch by batch.

---

##  Features

- **Hybrid Drift Detector** — combines performance monitoring, feature distribution tracking, and slope analysis to detect both abrupt and gradual drift
- **Statistical Baseline** — Page-Hinkley test runs in parallel for comparison
- **RL Agent (DQN)** — a Deep Q-Network learns to choose the best adaptation action (boost LR, restore concept memory, reset feature reference, etc.) in response to drift signals
- **Drift-only zoom** — live graph focuses only on the window around detected drift, not the full history
- **Accuracy dip graph** — overlays both detectors' drift points on the accuracy curve
- **Animated drift alerts** — staggered alert cards appear as drift events are detected
- **Side-by-side comparison** — metrics and batch comparison table for both methods
- **Full CI/CD pipeline** — lint, test, Docker build & push, auto-deploy to Render on every push to `main`

---

##  Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  Upload CSV → Run Detection → View Results           │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     train_and_detect()     │
         │   Batch-by-batch loop      │
         └──────┬────────────┬────────┘
                │            │
   ┌────────────▼──┐    ┌────▼──────────────┐
   │ PageHinkley   │    │ UnifiedDriftDetector│
   │ (Statistical) │    │ + RLDriftAgent(DQN) │
   └───────────────┘    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Actions chosen:    │
                    │  HOLD / BOOST_LR /  │
                    │  ADAPTIVE_LR /      │
                    │  RESTORE_CONCEPT /  │
                    │  RESET_FEATURE_REF  │
                    └─────────────────────┘
```

---

##  Getting Started

### Run locally

```bash
# Clone the repo
git clone https://github.com/Pratheek-05/multiscale-drift-detector.git
cd multiscale-drift-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Run with Docker

```bash
docker pull pratheek05/hybrid-drift-detection:latest
docker run -p 8501:8501 pratheek05/hybrid-drift-detection:latest
```

---

##  Project Structure

```
multiscale-drift-detector/
├── app.py                        # Main Streamlit dashboard + all ML components
├── statistical_detector.py       # Page-Hinkley drift detector
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── .dockerignore
├── .gitignore
├── tests/
│   └── test_detectors.py         # Unit tests (10 tests, pytest)
└── .github/
    └── workflows/
        └── ci-cd.yml             # GitHub Actions CI/CD pipeline
```

---

##  RL Agent — How It Works

The DQN agent observes a **5-dimensional state** at every batch:

| State Feature | Description |
|---|---|
| `perf_change` | Performance delta between recent and previous windows |
| `rolling_acc` | Rolling average accuracy |
| `slope` | Linear trend of accuracy over last N batches |
| `feature_score` | Mean + variance shift in input features |
| `cooldown_active` | Whether the detector is in cooldown |

And chooses from **5 actions**:

| Action | Effect |
|---|---|
| `HOLD` | Do nothing |
| `BOOST_LR` | Spike learning rate — abrupt drift response |
| `ADAPTIVE_LR` | Scale LR by slope magnitude — gradual drift response |
| `RESTORE_CONCEPT` | Load saved model weights from concept memory |
| `RESET_FEATURE_REF` | Update feature distribution reference |

**Reward signal:** accuracy improvement (+), correct drift reaction (+0.5), recovery bonus (+1.0), false action penalty (-0.2).

---

##  Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=.
```

```
tests/test_detectors.py::TestPageHinkley::test_no_drift_on_stable_stream     PASSED
tests/test_detectors.py::TestPageHinkley::test_detects_drift_on_step_change  PASSED
tests/test_detectors.py::TestPageHinkley::test_reset_clears_state            PASSED
tests/test_detectors.py::TestPageHinkley::test_returns_bool                  PASSED
tests/test_detectors.py::TestSimpleClassifier::test_forward_pass_shape       PASSED
tests/test_detectors.py::TestSimpleClassifier::test_forward_pass_no_nan      PASSED
tests/test_detectors.py::TestRLDriftAgent::test_select_action_valid_range    PASSED
tests/test_detectors.py::TestRLDriftAgent::test_epsilon_decreases            PASSED
tests/test_detectors.py::TestRLDriftAgent::test_reward_positive_on_accuracy_gain PASSED
tests/test_detectors.py::TestRLDriftAgent::test_reward_negative_on_false_action  PASSED

10 passed in ~7s
```

---

##  CI/CD Pipeline

Every push to `main` triggers the full pipeline:

```
Push to main
    │
    ├── Lint & Syntax Check    (pyflakes + flake8)
    ├── Run Tests              (pytest, 10 tests)
    ├── Build & Push Docker    (pratheek05/hybrid-drift-detection:latest)
    └── Deploy to Render       (auto-deploy via webhook)
```

Pushes to other branches only run lint + tests (no deploy).

---

##  Dataset Format

Upload any CSV where:
- All columns except the last are **numeric features**
- The **last column** is the binary label (`0` or `1`)

The app handles preprocessing (StandardScaler) automatically.

---

##  Configuration

All parameters are adjustable from the sidebar:

| Parameter | Default | Description |
|---|---|---|
| Batch Size | 50 | Samples per batch |
| Abrupt Threshold | -0.10 | Performance drop to trigger abrupt drift |
| Gradual Slope Threshold | -0.002 | Accuracy slope to trigger gradual drift |
| Feature Drift Threshold | 0.25 | Feature distribution shift magnitude |
| Performance Window | 15 | Batches to compare for performance change |
| Rolling Window | 30 | Batches for rolling accuracy |
| Cooldown Period | 30 | Batches to wait after a drift event |
| RL Learning Rate | 0.001 | DQN optimizer learning rate |
| RL Gamma | 0.95 | Discount factor for future rewards |
| Epsilon Decay Steps | 300 | Steps for exploration → exploitation |

---

##  Tech Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| Deep Learning | PyTorch |
| Data Processing | NumPy, Pandas, scikit-learn |
| Visualisation | Plotly |
| Statistical Detector | Page-Hinkley (custom) |
| RL Agent | DQN (custom PyTorch) |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Registry | Docker Hub |

---

##  Team

Built as part of a research project on adaptive machine learning systems.

---

<div align="center">
Made by Pratheek· <a href="https://github.com/Pratheek-05/multiscale-drift-detector">GitHub</a> · <a href="https://hub.docker.com/r/pratheek05/hybrid-drift-detection">Docker Hub</a>
</div>
