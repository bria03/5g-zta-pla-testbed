# 5G ZTA + PLA Testbed

This is a fully software-defined testbed demonstrating an integration of Zero Trust Architecture (ZTA) and Physical Layer Authentication (PLA) for use in private 5G networks.

The pipeline demonstrates stolen credential resilience: rogue devices with valid credentials are blocked at the physical layer before credential evaluation ever occurs.

---

## Architecture Overview

The testbed implements a two-stage security pipeline:

1. **PLA Stage** — Synthetic 5G NR SRS waveforms are generated in MATLAB, channel features are extracted pre-equalization, and an SVM classifier produces a continuous 0–1 trust score per device
2. **ZTA Stage** — OPA evaluates the trust score (TSA) and four UCA context attributes against a Rego policy, rendering one of three outcomes: `UCA_GRANT`, `UCA_DENY`, or `TSA_DENY`

MATLAB and Open5GS/UERANSIM run as parallel pipelines. The Python ZTA bridge connects them artificially — this is a known simulation constraint. In a physical deployment, trust scores would derive from live SRS captures at the gNodeB.

---

## Software Stack

| Component | Tool | Version |
|---|---|---|
| 5G Core | Open5GS | Latest (PPA) |
| RAN | srsRAN Project | ZMQ mode |
| UE Simulator | UERANSIM | v3.2.8 |
| Signal Generation | MATLAB + 5G Toolbox | R2024 |
| ML Classifier | Python / scikit-learn | 1.x |
| ZTA Policy Engine | Open Policy Agent | Latest binary |
| Identity Provider | Keycloak | 26.x dev mode |
| Metrics | Prometheus + Grafana | apt stable |
| OS | Ubuntu 24.04 LTS (WSL2) | x86_64 |

---

## Simulated Devices

Four devices are simulated — two legitimate, two rogue. Hardware imperfection parameters (CFO, IQ imbalance, phase noise) are sampled from Gaussian distributions per transmission to model natural per-device RF hardware drift. Device 3 (Rogue) is intentionally placed close to Device 2 (Legitimate) in feature space to stress-test the SVM decision boundary.

| Device | Label | Avg Trust Score (n=500 enrollment) |
|---|---|---|
| Device 1 | Legitimate | 0.9667 ± 0.0918 |
| Device 2 | Legitimate | 0.9313 ± 0.0461 |
| Device 3 | Rogue | 0.1308 ± 0.1664 |
| Device 4 | Rogue | 0.0743 ± 0.2129 |

Overall SVM classifier accuracy at n=500 enrollment samples: **97%**

---

## Simulation Scope & Constraints

- **Setup A** — fully software-defined, no physical radio hardware
- Channel model: MATLAB TDL-C stochastic model (not ray-traced)
- Trust scores in pipeline are static mean vectors, not live per-transmission captures
- Per-transmission variance partially modeled via Gaussian sampling (n=20) in the ZTA bridge
- UCA context attributes partially simulated — `access_hour` is real system clock, `requested_slice` and `request_frequency` are statically configured per device
- No continuous ZTA re-evaluation or feedback loop retraining implemented
- Sionna RT ray-traced channel and deep learning RF fingerprinting (CNN/LSTM on raw IQ) are identified as future work contingent on CUDA GPU access

---

## Installation

Follow the full install guide: [`docs/install_guide.html`](docs/install_guide.html)

Covers: MongoDB, Open5GS, srsRAN Project (ZMQ build), Python ML stack, OPA, Keycloak, Prometheus, Grafana — all on Ubuntu 24.04 LTS.

MATLAB + 5G Toolbox must be separately licensed and installed on Windows. The MATLAB script writes feature output to the WSL2 shared path accessible from Ubuntu.

---

## Running the Pipeline

Follow the startup and shutdown guide: [`docs/startup_shutdown.html`](docs/startup_shutdown.html)

Quick sequence:
1. Verify Open5GS services and restore ogstun interface
2. Start OPA server and reload Rego policy
3. Start Keycloak
4. Start srsRAN Project gNB (Terminal 1)
5. Start UERANSIM gNB (Terminal 2)
6. Start UERANSIM UE (Terminal 3)
7. Start PLA Prometheus exporter
8. Run `python3 python/zta_bridge.py` for a full end-to-end evaluation

---

## Credentials Notice

`config/ueransim/ue.yaml` contains IMSI, K, and OPc values. These are standard test credentials (MCC 999, test PLMN) and do not represent real subscriber data. `python/zta_bridge.py` ships with placeholder values for `CLIENT_SECRET` and device passwords — replace these with your own Keycloak realm credentials before running.

---

## Repository Structure

```
5g-zta-pla-testbed/
├── README.md
├── matlab/
│   └── generate_pla_features.m
├── python/
│   ├── pla_classifier.py
│   ├── zta_bridge.py
│   └── pla_exporter.py
├── opa/
│   └── zta_policy.rego
├── config/
│   ├── gnb_zmq.yaml
│   ├── open5gs/
│   │   ├── amf.yaml
│   │   └── upf.yaml
│   └── ueransim/
│       ├── gnb.yaml
│       └── ue.yaml
└── docs/
    ├── install_guide.html
    ├── startup_shutdown.html
    ├── simulation_summary_v2.html
    └── testbed_flowchart.html
```
