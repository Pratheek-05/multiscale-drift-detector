import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CONFIGURATION
# ==========================================================

dataset = "gradual.csv"  # Change dataset here
batch_size = 50

base_lr = 0.001
max_lr = 0.005
alpha = 5.0

rolling_window = 30
negative_slope_threshold = -0.002
min_slope_magnitude = 0.003   # prevents tiny noise triggers

feature_drift_threshold = 0.25   # slightly increased for stability
cooldown_period = 25

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(dataset)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

total_samples = len(X)
total_batches = total_samples // batch_size

print(f"\nRunning Linear Drift Adaptive Model (Stable v3) on {dataset}")
print("\n================ DATASET INFO ================")
print(f"Total Samples : {total_samples}")
print(f"Batch Size    : {batch_size}")
print(f"Total Batches : {total_batches}")
print("==============================================")

# ==========================================================
# MODEL
# ==========================================================

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=base_lr)

# ==========================================================
# TRACKING VARIABLES
# ==========================================================

accuracy_history = []
rolling_accuracy_history = []
learning_rate_history = []

feature_mean_ref = None
feature_var_ref = None

last_adaptation_batch = -100

# ==========================================================
# TRAINING LOOP
# ==========================================================

for batch_idx in range(total_batches):

    start = batch_idx * batch_size
    end = start + batch_size

    X_batch = X[start:end]
    y_batch = y[start:end]

    X_tensor = torch.FloatTensor(X_batch)
    y_tensor = torch.FloatTensor(y_batch).view(-1, 1)

    # ======================================================
    # FEATURE DRIFT MONITORING
    # ======================================================

    current_mean = np.mean(X_batch, axis=0)
    current_var = np.var(X_batch, axis=0)

    if feature_mean_ref is None:
        feature_mean_ref = current_mean
        feature_var_ref = current_var

    mean_shift = np.linalg.norm(current_mean - feature_mean_ref)
    var_shift = np.linalg.norm(current_var - feature_var_ref)

    feature_drift_score = mean_shift + var_shift
    feature_drift_detected = feature_drift_score > feature_drift_threshold

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    # ======================================================
    # ACCURACY
    # ======================================================

    predictions = (outputs.detach().numpy() > 0.5).astype(int)
    accuracy = np.mean(predictions == y_batch.reshape(-1, 1))
    accuracy_history.append(accuracy)

    # ======================================================
    # ROLLING ACCURACY
    # ======================================================

    if len(accuracy_history) >= rolling_window:
        rolling_acc = np.mean(accuracy_history[-rolling_window:])
    else:
        rolling_acc = np.mean(accuracy_history)

    rolling_accuracy_history.append(rolling_acc)

    # ======================================================
    # SLOPE CALCULATION
    # ======================================================

    if len(rolling_accuracy_history) >= rolling_window:
        y_vals = rolling_accuracy_history[-rolling_window:]
        x_vals = np.arange(len(y_vals))
        slope = np.polyfit(x_vals, y_vals, 1)[0]
    else:
        slope = 0

    performance_drift = (
        slope < negative_slope_threshold and
        abs(slope) > min_slope_magnitude
    )

    # ======================================================
    # DRIFT CONFIDENCE (STRICT MODE)
    # ======================================================

    drift_confidence = 0

    if performance_drift:
        drift_confidence += 1

    if feature_drift_detected:
        drift_confidence += 1

    strong_drift = drift_confidence >= 2   # BOTH signals required

    # ======================================================
    # ADAPTIVE LEARNING RATE WITH COOLDOWN
    # ======================================================

    current_lr = base_lr

    if strong_drift and (batch_idx - last_adaptation_batch > cooldown_period):

        drift_strength = abs(slope)
        scaled_lr = base_lr * (1 + alpha * drift_strength)
        scaled_lr = min(scaled_lr, max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = scaled_lr

        current_lr = scaled_lr
        last_adaptation_batch = batch_idx

        # Reset feature reference after confirmed drift
        feature_mean_ref = current_mean
        feature_var_ref = current_var

        print(f"Drift adaptation at batch {batch_idx} | Confidence: {drift_confidence}")

    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr
        current_lr = base_lr

    learning_rate_history.append(current_lr)

print("\nExecution completed successfully.")

# ==========================================================
# PLOTTING
# ==========================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(accuracy_history)
plt.title("Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")

plt.subplot(1, 3, 2)
plt.plot(rolling_accuracy_history)
plt.title("Rolling Accuracy")
plt.xlabel("Batch")
plt.ylabel("Rolling Accuracy")

plt.subplot(1, 3, 3)
plt.plot(learning_rate_history)
plt.title("Learning Rate")
plt.xlabel("Batch")
plt.ylabel("LR")

plt.tight_layout()
plt.show()