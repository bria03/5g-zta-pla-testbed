import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load feature data
df = pd.read_csv('/home/bdeloatch/pla_data/features.csv')

print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['Label'].value_counts()}")
print(f"Device distribution:\n{df['DeviceID'].value_counts()}")

X = df[['CFO','IQImbalance','RxPower','PhaseVar','AmpVar','CFOVar']].values
y = df['Label'].values
device_ids = df['DeviceID'].values

# Split enrollment vs test per device (50 enroll, 20 test each)
# Device 1: rows 0-69,    Device 2: rows 70-139
# Device 3: rows 140-209, Device 4: rows 210-279
enroll_idx = (list(range(0,100))   + list(range(200,300)) +
              list(range(400,500)) + list(range(600,700)))
test_idx   = (list(range(100,200))  + list(range(300,400)) +
              list(range(500,600)) + list(range(700,800)))

X_train = X[enroll_idx]
y_train = y[enroll_idx]
X_test  = X[test_idx]
y_test  = y[test_idx]
dev_test = device_ids[test_idx]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train SVM
svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Evaluate overall
y_pred = svm.predict(X_test_scaled)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred,
      target_names=['Rogue UE', 'Legitimate UE']))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Trust scores
trust_scores = svm.predict_proba(X_test_scaled)[:, 1]

# Per-device trust score breakdown
print("\n=== Per-Device Trust Scores ===")
for d in [1, 2, 3, 4]:
    mask = dev_test == d
    avg = np.mean(trust_scores[mask])
    label = 'Legitimate' if y_test[mask][0] == 1 else 'Rogue'
    print(f"Device {d} ({label}): avg trust score = {avg:.4f}")

# FAR/FRR curve
fpr, tpr, thresholds = roc_curve(y_test, trust_scores)
far = fpr
frr = 1 - tpr

plt.figure(figsize=(8,5))
plt.plot(thresholds, far[:len(thresholds)], label='FAR (False Acceptance)')
plt.plot(thresholds, frr[:len(thresholds)], label='FRR (False Rejection)')
plt.xlabel('Threshold T')
plt.ylabel('Error Rate')
plt.title('PLA Classifier — FAR/FRR Tradeoff')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/bdeloatch/pla_data/far_frr_curve.png', dpi=300)
print("\nFAR/FRR curve saved to pla_data/far_frr_curve.png\n")

# Save model
joblib.dump(svm,    '/home/bdeloatch/pla_data/svm_model.pkl')
joblib.dump(scaler, '/home/bdeloatch/pla_data/scaler.pkl')
print(r"Model and scaler saved to \\wsl.localhost\Ubuntu\home\bdeloatch\pla_data")
print("\n")
