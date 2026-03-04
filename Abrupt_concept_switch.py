import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import copy
import random

# =====================================================
# REPRODUCIBILITY
# =====================================================

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================================================
# SELECT DATASET
# =====================================================

dataset = "abrupt.csv"   # Change dataset here

# =====================================================
# MODEL
# =====================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# PERFORMANCE MONITOR
# =====================================================

class PerformanceMonitor:
    def __init__(self, window=15):
        self.window = window
        self.history = []

    def update(self, accuracy):
        self.history.append(accuracy)

        if len(self.history) < self.window * 2:
            return 0

        recent = np.mean(self.history[-self.window:])
        previous = np.mean(self.history[-2*self.window:-self.window])
        return recent - previous

# =====================================================
# LOAD DATA
# =====================================================

print(f"\nRunning Stabilized Abrupt Concept Switching on {dataset}")

df = pd.read_csv(dataset)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

input_dim = X.shape[1]
batch_size = 50

total_samples = len(X)
total_batches = int(np.ceil(total_samples / batch_size))

print("\n================ DATASET INFO ================")
print(f"Total Samples : {total_samples}")
print(f"Batch Size    : {batch_size}")
print(f"Total Batches : {total_batches}")
print("==============================================\n")

# =====================================================
# INITIALIZATION
# =====================================================

model = SimpleClassifier(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

monitor = PerformanceMonitor()

concept_memory = {}
current_concept_id = 0

accuracy_history = []
concept_history = []

NEGATIVE_THRESHOLD = -0.10
POSITIVE_THRESHOLD = 0.10

# 🔵 COOLDOWN SETTINGS
cooldown_period = 40
cooldown_counter = 0

# =====================================================
# STREAMING LOOP
# =====================================================

for batch_idx, i in enumerate(range(0, len(X), batch_size)):

    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]

    X_tensor = torch.tensor(X_batch, dtype=torch.float32)
    y_tensor = torch.tensor(y_batch, dtype=torch.long)

    # ---- Evaluate BEFORE training ----
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_tensor).float().mean().item()

    accuracy_history.append(acc)

    slope = monitor.update(acc)

    # =====================================================
    # STABILIZED CONCEPT SWITCHING
    # =====================================================

    if cooldown_counter > 0:
        cooldown_counter -= 1

    else:
        # 🔴 Strong Negative Slope → New Concept
        if slope < NEGATIVE_THRESHOLD:

            print(f"Drift detected at batch {batch_idx}")

            concept_memory[current_concept_id] = copy.deepcopy(model.state_dict())

            current_concept_id = 1 - current_concept_id
            optimizer = optim.Adam(model.parameters(), lr=0.003)

            cooldown_counter = cooldown_period

        # 🟢 Strong Positive Slope → Reload Stored Concept
        elif slope > POSITIVE_THRESHOLD:

            if current_concept_id in concept_memory:

                print(f"Reloading stored concept at batch {batch_idx}")

                model.load_state_dict(concept_memory[current_concept_id])
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                cooldown_counter = cooldown_period

    concept_history.append(current_concept_id)

    # ---- Train ----
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# =====================================================
# METRICS & VISUALIZATION
# =====================================================

def moving_average(data, window=30):
    return np.convolve(data, np.ones(window)/window, mode='same')

performance = np.array(accuracy_history) * 1000
smoothed_perf = moving_average(performance, 30)
rolling_acc = moving_average(accuracy_history, 20)

# =====================================================
# PLOTS (STABLE VERSION)
# =====================================================

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

axs[0, 0].plot(smoothed_perf)
axs[0, 0].set_xlabel("Batch")
axs[0, 0].set_ylabel("Performance")

axs[0, 1].plot(accuracy_history)
axs[0, 1].set_xlabel("Batch")
axs[0, 1].set_ylabel("Accuracy")

axs[1, 0].plot(rolling_acc)
axs[1, 0].set_xlabel("Batch")
axs[1, 0].set_ylabel("Rolling Accuracy")

axs[1, 1].plot(concept_history)
axs[1, 1].set_xlabel("Batch")
axs[1, 1].set_ylabel("Concept ID")

plt.tight_layout()
plt.show()

print("\nExecution completed successfully.")
