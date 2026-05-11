import time
import json
import os
from prometheus_client import start_http_server, Gauge, Counter
import joblib
import numpy as np
import pandas as pd

# Load trained model
svm = joblib.load('/home/bdeloatch/pla_data/svm_model.pkl')
scaler = joblib.load('/home/bdeloatch/pla_data/scaler.pkl')

# Load feature data for live scoring
df = pd.read_csv('/home/bdeloatch/pla_data/features.csv')

DEVICE_FEATURES = {}
for device_id in [1, 2, 3, 4]:
    device_rows = df[df['DeviceID'] == device_id]
    enroll_rows = device_rows.head(50)
    mean_features = enroll_rows[['CFO','IQImbalance','RxPower',
                                  'PhaseVar','AmpVar','CFOVar']].mean().values
    DEVICE_FEATURES[device_id] = mean_features

# Prometheus metrics
trust_score_gauge = Gauge(
    'pla_trust_score',
    'PLA trust score per device',
    ['device_id', 'label']
)

decision_counter = Counter(
    'zta_decision_total',
    'ZTA access decisions',
    ['device_id', 'decision']
)

tsa_threshold_gauge = Gauge(
    'pla_tsa_threshold',
    'Current TSA threshold value'
)

# Device labels
DEVICE_LABELS = {1: 'legitimate', 2: 'legitimate', 3: 'rogue', 4: 'rogue'}

THRESHOLD = 0.50

def compute_trust_scores():
    """Compute and expose trust scores for all devices."""
    tsa_threshold_gauge.set(THRESHOLD)

    for device_id in [1, 2, 3, 4]:
        features = DEVICE_FEATURES[device_id].reshape(1, -1)
        features_scaled = scaler.transform(features)
        trust_score = svm.predict_proba(features_scaled)[0][1]
        label = DEVICE_LABELS[device_id]

        trust_score_gauge.labels(
            device_id=str(device_id),
            label=label
        ).set(trust_score)

def read_audit_log():
    """Read latest decisions from audit log and update counters."""
    audit_path = '/home/bdeloatch/pla_data/audit_log.jsonl'
    if not os.path.exists(audit_path):
        return

    with open(audit_path, 'r') as f:
        lines = f.readlines()

    # Only process last 4 entries (one per device per bridge run)
    for line in lines[-4:]:
        entry = json.loads(line.strip())
        decision_counter.labels(
            device_id=str(entry['device_id']),
            decision=entry['decision']
        ).inc()

if __name__ == "__main__":
    # Start Prometheus metrics server on port 9200
    # (avoiding 9090/9091 used by Open5GS and Prometheus)
    start_http_server(9200)
    print("PLA Prometheus exporter running on port 9200")
    print("Metrics available at http://localhost:9200/metrics")

    while True:
        compute_trust_scores()
        read_audit_log()
        time.sleep(15)