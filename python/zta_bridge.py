import requests
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Load trained PLA model
svm = joblib.load('/home/bdeloatch/pla_data/svm_model.pkl')
scaler = joblib.load('/home/bdeloatch/pla_data/scaler.pkl')

# Keycloak config
KEYCLOAK_URL = "http://localhost:8080"
REALM = "private5G"
CLIENT_ID = "zta-client"
CLIENT_SECRET = "INSERT CLIENT SECRET HERE"

# OPA config
OPA_URL = "http://localhost:8181/v1/data/zta/access"

# Device registry - maps SUPI to device profile and Keycloak username
DEVICE_REGISTRY = {
    "999700000000001": {"device_id": 1, "username": "ue-device-1", "password": "INSERT DEVICE PASSWORD HERE", "label": "Legitimate"},
    "999700000000002": {"device_id": 2, "username": "ue-device-2", "password": "INSERT DEVICE PASSWORD HERE", "label": "Legitimate"},
    "999700000000003": {"device_id": 3, "username": "ue-device-3", "password": "INSERT DEVICE PASSWORD HERE", "label": "Rogue"},
    "999700000000004": {"device_id": 4, "username": "ue-device-4", "password": "INSERT DEVICE PASSWORD HERE", "label": "Rogue"},
}

# Simulated feature vectors per device - in full pipeline these come from MATLAB CSV
# Using representative values from classifier output
# Load actual feature data and compute per-device mean vectors
df = pd.read_csv('/home/bdeloatch/pla_data/features.csv')

DEVICE_FEATURES = {}
for device_id in [1, 2, 3, 4]:
    device_rows = df[df['DeviceID'] == device_id]
    # Use enrollment samples only (first 50 rows per device)
    enroll_rows = device_rows.head(50)
    mean_features = enroll_rows[['CFO','IQImbalance','RxPower',
                                  'PhaseVar','AmpVar','CFOVar']].mean().values
    DEVICE_FEATURES[device_id] = mean_features

def get_keycloak_token(username, password):
    """Get OAuth2 token from Keycloak for UE identity verification."""
    response = requests.post(
        f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "username": username,
            "password": password
        }
    )
    if response.status_code == 200:
        return response.json().get("access_token")
    return None

def get_pla_trust_score(device_id, n_samples=20):
    """
    Get PLA trust score with per-transmission variance.
    Simulates natural per-transmission hardware drift by sampling
    n_samples feature vectors from the device's learned distribution
    and returning mean +/- std of resulting trust scores.
    """
    # Get enrollment rows for this device
    device_rows = df[df['DeviceID'] == device_id].head(100)
    
    # Compute per-feature mean and std from enrollment samples
    feature_cols = ['CFO','IQImbalance','RxPower','PhaseVar','AmpVar','CFOVar']
    feat_mean = device_rows[feature_cols].mean().values
    feat_std  = device_rows[feature_cols].std().values
    
    # Sample n_samples feature vectors from device's learned distribution
    samples = np.random.normal(
        loc=feat_mean,
        scale=feat_std,
        size=(n_samples, len(feature_cols))
    )
    
    # Score all samples
    samples_scaled = scaler.transform(samples)
    scores = svm.predict_proba(samples_scaled)[:, 1]
    
    mean_score = round(float(np.mean(scores)), 4)
    std_score  = round(float(np.std(scores)), 4)
    
    return mean_score, std_score
	
def get_uca_context(device_id):
    """Get UCA context attributes per device."""
    contexts = {
        1: {"access_hour": datetime.now().hour, "requested_slice": 1, "request_frequency": 5},   # all pass
        2: {"access_hour": datetime.now().hour, "requested_slice": 99, "request_frequency": 5},  # invalid slice — UCA_DENY
        3: {"access_hour": datetime.now().hour, "requested_slice": 1, "request_frequency": 5},   # TSA_DENY anyway
        4: {"access_hour": datetime.now().hour, "requested_slice": 1, "request_frequency": 5},   # TSA_DENY anyway
    }
    return contexts[device_id]

def evaluate_zta(supi):
    """Run full ZTA evaluation pipeline for a given SUPI."""
    if supi not in DEVICE_REGISTRY:
        return {"error": f"SUPI {supi} not registered"}

    device = DEVICE_REGISTRY[supi]
    device_id = device["device_id"]

    print(f"\n{'='*55}")
    print(f"ZTA Evaluation - SUPI: {supi} ({device['label']} UE)")
    print(f"{'='*55}")

    # Step 1 — PLA trust score
    trust_score, trust_std = get_pla_trust_score(device_id)
    print(f"[PLA]  Trust Score: {trust_score:.4f} ± {trust_std:.4f} (n=20 samples)")

    # Step 2 - Keycloak token (identity verification)
    token = get_keycloak_token(device["username"], device["password"])
    identity_verified = token is not None
    print(f"[KC]   Identity Verified: {identity_verified}")

    # Step 3 - UCA context
    context = get_uca_context(device_id)
    print(f"[UCA]  Context: hour={context['access_hour']}, "
          f"slice={context['requested_slice']}, "
          f"freq={context['request_frequency']}")

    # Step 4 - OPA policy evaluation
    opa_input = {
        "input": {
            "supi": supi,
            "trust_score": trust_score,
            "access_hour": context["access_hour"],
            "requested_slice": context["requested_slice"],
            "request_frequency": context["request_frequency"]
        }
    }

    opa_response = requests.post(OPA_URL, json=opa_input)
    result = opa_response.json().get("result", {})

    decision = result.get("decision", "UNKNOWN")
    allow = result.get("allow", False)

    print(f"[OPA]  Decision: {decision}")
    print(f"[OPA]  Access: {'GRANTED' if allow else 'DENIED'}")

    # Log full result
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "supi": supi,
        "device_id": device_id,
        "label": device["label"],
        "trust_score": trust_score,
        "trust_score_std": trust_std,
        "identity_verified": identity_verified,
        "context": context,
        "decision": decision,
        "allow": allow
    }

    # Append to audit log
    with open('/home/bdeloatch/pla_data/audit_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    return log_entry

# Run evaluation for all four devices
if __name__ == "__main__":
    supis = [
        "999700000000001",
        "999700000000002",
        "999700000000003",
        "999700000000004"
    ]

    results = []
    for supi in supis:
        result = evaluate_zta(supi)
        results.append(result)

    print(f"\n{'='*55}")
    print("PIPELINE SUMMARY")
    print(f"{'='*55}")
    for r in results:
        print(f"Device {r['device_id']} ({r['label']:10}) | "
              f"Trust: {r['trust_score']:.4f}  ± {r['trust_score_std']:.4f} "
              f"Decision: {r['decision']}")
    print("\n")
