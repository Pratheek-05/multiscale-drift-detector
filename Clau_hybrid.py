# ============================================================
# Hybrid Drift Adaptive System v3.3 - TUNED
# Reduced False Positives + Better Drift Tracking
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque
import random

# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

dataset = "gradual.csv"  # Change dataset here

# ============================================================
# MODEL
# ============================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# LR CONTROLLER
# ============================================================

class LearningRateController:
    def __init__(self, optimizer, base_lr=0.001, max_lr=0.01, decay=0.98):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.decay = decay
        self.steps_since_increase = 0

    def increase(self, factor):
        current_lr = self.optimizer.param_groups[0]['lr']
        new_lr = min(self.max_lr, current_lr * factor)
        self.optimizer.param_groups[0]['lr'] = new_lr
        self.steps_since_increase = 0
        print(f"  → LR increased to {new_lr:.6f}")

    def decay_back(self):
        self.steps_since_increase += 1
        
        # Wait at least 10 steps after increase before decaying
        if self.steps_since_increase < 10:
            return
            
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr > self.base_lr:
            new_lr = max(self.base_lr, current_lr * self.decay)
            self.optimizer.param_groups[0]['lr'] = new_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# ============================================================
# HYBRID DRIFT ENGINE v3.3 - TUNED FOR REDUCED FALSE POSITIVES
# ============================================================

class HybridDriftEngineV3_3:
    """
    KEY IMPROVEMENTS from v3.2:
    - STRICTER thresholds (2.0 std instead of 1.5)
    - Longer confirm window (5 instead of 3)
    - BLOCKS re-detection during active drift
    - Better severity calculation
    - Cooldown period after recovery
    """

    def __init__(self, confirm_window=5, slope_window=10, warmup=15):
        self.prev_acc = None
        self.confirm_counter = 0.0
        self.confirm_window = confirm_window  # Increased from 3 to 5
        self.warmup = warmup  # Increased from 10 to 15

        self.acc_deltas = deque(maxlen=100)  # Longer history
        self.feature_shifts = deque(maxlen=100)
        self.slope_window = deque(maxlen=slope_window)  # Increased from 7 to 10

        self.in_drift = False
        self.pre_drift_peak = 0
        self.drift_start = None
        self.recovery_times = []
        self.drift_log = []
        
        # Feature EMA tracking
        self.feature_ema = None
        self.ema_alpha = 0.3
        
        # NEW: Cooldown to prevent re-detection
        self.cooldown_counter = 0
        self.cooldown_period = 15  # Batches to wait after drift detection
        
        # Stats
        self.total_batches = 0

    def detect(self, batch, rolling_acc, feature_shift, current_feature_mean=None):
        """
        TUNED drift detection with stricter thresholds
        """
        self.total_batches += 1
        
        # Cooldown management
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            # Still update tracking variables during cooldown
            if self.prev_acc is None:
                self.prev_acc = rolling_acc
            else:
                self.prev_acc = rolling_acc
                delta_acc = self.prev_acc - rolling_acc
                slope = rolling_acc - self.prev_acc
                self.acc_deltas.append(delta_acc)
                self.feature_shifts.append(feature_shift)
                self.slope_window.append(slope)
            
            # Update EMA
            if current_feature_mean is not None:
                if self.feature_ema is None:
                    self.feature_ema = current_feature_mean
                else:
                    self.feature_ema = (self.ema_alpha * current_feature_mean + 
                                       (1 - self.ema_alpha) * self.feature_ema)
            
            return None  # No detection during cooldown
        
        # WARMUP PERIOD
        if batch < self.warmup:
            if self.prev_acc is None:
                self.prev_acc = rolling_acc
                self.pre_drift_peak = rolling_acc
                if current_feature_mean is not None:
                    self.feature_ema = current_feature_mean
            else:
                self.prev_acc = rolling_acc
                self.pre_drift_peak = max(self.pre_drift_peak, rolling_acc)
                
                if current_feature_mean is not None and self.feature_ema is not None:
                    self.feature_ema = (self.ema_alpha * current_feature_mean + 
                                       (1 - self.ema_alpha) * self.feature_ema)
            return None

        # Track peak BEFORE drift
        if not self.in_drift:
            if rolling_acc > self.pre_drift_peak:
                self.pre_drift_peak = rolling_acc

        # Calculate metrics
        delta_acc = self.prev_acc - rolling_acc
        slope = rolling_acc - self.prev_acc

        self.acc_deltas.append(delta_acc)
        self.feature_shifts.append(feature_shift)
        self.slope_window.append(slope)

        # STRICTER: 2.0 std instead of 1.5 std
        acc_std = np.std(self.acc_deltas) if len(self.acc_deltas) > 10 else 0.05
        feat_std = np.std(self.feature_shifts) if len(self.feature_shifts) > 10 else 0.1
        
        # Higher minimum thresholds
        acc_threshold = max(0.05, np.mean(self.acc_deltas) + 2.0 * acc_std)  # 2.0x
        feat_threshold = max(0.15, np.mean(self.feature_shifts) + 2.0 * feat_std)  # 2.0x

        # PRIMARY CONDITION: More strict
        primary_condition = (delta_acc > acc_threshold) and \
                           (feature_shift > feat_threshold) and \
                           (delta_acc > 0.01)  # Additional floor

        # SLOPE DETECTION: Require 80% negative (stricter)
        negative_slopes = sum(1 for s in self.slope_window if s < -0.002)
        continuous_drop = negative_slopes >= (len(self.slope_window) * 0.8)  # 80%

        # Confirm counter with slower decay
        if primary_condition or continuous_drop:
            self.confirm_counter += 1
        else:
            self.confirm_counter = max(0, self.confirm_counter - 0.3)  # Slower decay

        drift_event = None

        # Drift confirmed (higher threshold)
        if self.confirm_counter >= self.confirm_window:

            drift_type = self.classify(delta_acc, continuous_drop, primary_condition)
            severity = self.compute_severity(delta_acc, feature_shift)

            drift_event = {
                "batch": batch,
                "type": drift_type,
                "severity": round(severity, 3),
                "confidence": round(self.confirm_counter, 1),
                "delta_acc": round(delta_acc, 4),
                "feature_shift": round(feature_shift, 4)
            }

            # NEW: Only log if NOT already in drift
            if not self.in_drift:
                self.drift_start = batch
                self.in_drift = True
                self.drift_log.append(drift_event)
                
                # Set cooldown to prevent immediate re-detection
                self.cooldown_counter = self.cooldown_period
            
            self.confirm_counter = 0.0

        # Recovery logic
        if self.in_drift and self.drift_start is not None:
            target = 0.95 * self.pre_drift_peak
            
            if rolling_acc >= target:
                recovery = batch - self.drift_start
                self.recovery_times.append(recovery)
                
                print(f"  Recovered after {recovery} batches "
                      f"(target: {target:.3f}, current: {rolling_acc:.3f})")
                
                # Reset drift state
                self.in_drift = False
                self.drift_start = None
                self.pre_drift_peak = rolling_acc
                
                # NEW: Set cooldown after recovery
                self.cooldown_counter = 10

        # Update tracking
        self.prev_acc = rolling_acc
        
        # Update feature EMA
        if current_feature_mean is not None:
            if self.feature_ema is None:
                self.feature_ema = current_feature_mean
            else:
                self.feature_ema = (self.ema_alpha * current_feature_mean + 
                                   (1 - self.ema_alpha) * self.feature_ema)

        return drift_event

    def classify(self, delta_acc, continuous_drop, primary_condition):
        """Proper drift type classification"""
        # Abrupt: Large sudden drop
        if delta_acc > 0.2 and primary_condition:
            return "Abrupt Drift"
        
        # Linear: Continuous moderate decline
        elif continuous_drop and delta_acc > 0.05:
            return "Linear Drift"
        
        # Gradual: Slow continuous decline
        elif continuous_drop:
            return "Gradual Drift"
        
        # Feature-driven drift
        elif primary_condition:
            return "Feature Drift"
        
        else:
            return "Minor Drift"

    def compute_severity(self, delta_acc, feature_shift):
        """Enhanced severity calculation"""
        # Normalize with higher caps
        acc_component = min(delta_acc / 0.4, 1.0)  # Higher normalization
        feat_component = min(feature_shift / 2.0, 1.0)  # Higher normalization
        
        # Weight accuracy more heavily
        severity = 0.8 * acc_component + 0.2 * feat_component
        return max(0.0, min(1.0, severity))

    def get_metrics(self):
        """Compute comprehensive metrics"""
        return {
            "total_drifts": len(self.drift_log),
            "avg_recovery": round(np.mean(self.recovery_times), 2) 
                           if self.recovery_times else None,
            "median_recovery": round(np.median(self.recovery_times), 2)
                              if self.recovery_times else None,
            "max_recovery": max(self.recovery_times) if self.recovery_times else None,
            "drift_types": {dt["type"]: sum(1 for d in self.drift_log if d["type"] == dt["type"]) 
                           for dt in self.drift_log} if self.drift_log else {},
            "avg_severity": round(np.mean([d["severity"] for d in self.drift_log]), 3)
                           if self.drift_log else None,
            "currently_in_drift": self.in_drift,
            "false_positive_rate": self._estimate_false_positives()
        }

    def _estimate_false_positives(self):
        """Estimate false positive rate based on recovery times"""
        if not self.recovery_times:
            return 0.0
        
        # Drifts that "recover" in 0-2 batches are likely false positives
        fast_recoveries = sum(1 for r in self.recovery_times if r <= 2)
        return round(fast_recoveries / len(self.recovery_times) * 100, 1)

    def get_drift_summary(self):
        """Pretty print drift log"""
        if not self.drift_log:
            return "No drifts detected"
        
        summary = "\n=== DRIFT DETECTION LOG ===\n"
        for i, drift in enumerate(self.drift_log, 1):
            summary += (f"{i}. Batch {drift['batch']}: {drift['type']} "
                       f"(severity={drift['severity']}, conf={drift['confidence']})\n")
        return summary

# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv(dataset)
    print(f" Loaded dataset: {dataset}")
    print(f" Shape: {df.shape}")
except FileNotFoundError:
    print(f" Dataset '{dataset}' not found. Creating synthetic data for demo...")
    np.random.seed(42)
    n = 5000
    X1 = np.random.randn(n//2, 5)
    y1 = (X1[:, 0] + X1[:, 1] > 0).astype(int)
    
    X2 = np.random.randn(n//2, 5) + 2.0
    y2 = (X2[:, 0] - X2[:, 1] > 0).astype(int)
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    df = pd.DataFrame(X)
    df['label'] = y
    print("  Created synthetic drift dataset")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

batch_size = 50
total_batches = len(X) // batch_size

print(f"  Total batches: {total_batches}")
print(f"  Batch size: {batch_size}\n")

# ============================================================
# TRAINING SETUP
# ============================================================

model = SimpleClassifier(X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

lr_controller = LearningRateController(optimizer)
engine = HybridDriftEngineV3_3(
    confirm_window=5,    # Stricter: need 5 confirmations
    slope_window=10,     # Longer trend window
    warmup=15           # Longer warmup
)

rolling_window = deque(maxlen=20)
rolling_acc_history = []
feature_shift_history = []
lr_history = []

prev_feature_mean = None

# ============================================================
# STREAMING LOOP
# ============================================================

print("=" * 60)
print("STARTING STREAMING TRAINING")
print("=" * 60)

for batch in range(total_batches):

    X_batch = X[batch*batch_size:(batch+1)*batch_size]
    y_batch = y[batch*batch_size:(batch+1)*batch_size]

    X_tensor = torch.tensor(X_batch, dtype=torch.float32)
    y_tensor = torch.tensor(y_batch, dtype=torch.long)

    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_tensor).float().mean().item()

    rolling_window.append(acc)
    rolling_acc = np.mean(rolling_window)
    rolling_acc_history.append(rolling_acc)

    current_mean = np.mean(X_batch, axis=0)
    if prev_feature_mean is None:
        feature_shift = 0
    else:
        feature_shift = np.linalg.norm(current_mean - prev_feature_mean)

    feature_shift_history.append(feature_shift)
    prev_feature_mean = current_mean

    drift = engine.detect(batch, rolling_acc, feature_shift, current_mean)

    if drift:
        print(f"\n DRIFT DETECTED at batch {batch}")
        print(f"  Type: {drift['type']}")
        print(f"  Severity: {drift['severity']}")
        print(f"  Confidence: {drift['confidence']}")
        print(f"  Acc: {drift['delta_acc']:.4f}, Feature: {drift['feature_shift']:.4f}")

        # Adaptive learning rate
        if drift["type"] == "Abrupt Drift":
            lr_controller.increase(2.0)
        elif drift["type"] in ["Linear Drift", "Gradual Drift"]:
            lr_controller.increase(1.5)
        else:
            lr_controller.increase(1.3)

    # Only decay when stable
    if drift is None and not engine.in_drift:
        lr_controller.decay_back()
    
    lr_history.append(lr_controller.get_lr())

    if batch % 20 == 0 and batch > 0:
        status = "IN DRIFT" if engine.in_drift else "STABLE"
        cooldown = f" (cooldown: {engine.cooldown_counter})" if engine.cooldown_counter > 0 else ""
        print(f"Batch {batch}/{total_batches} | "
              f"Acc: {rolling_acc:.3f} | "
              f"LR: {lr_controller.get_lr():.6f} | "
              f"{status}{cooldown}")

# ============================================================
# FINAL METRICS
# ============================================================

metrics = engine.get_metrics()

print("\n" + "=" * 60)
print("PRODUCTION METRICS - HYBRID DRIFT SYSTEM")
print("=" * 60)
print(f"Total Drifts Detected: {metrics['total_drifts']}")
print(f"Drift Types: {metrics['drift_types']}")
print(f"Average Severity: {metrics['avg_severity']}")
print(f"Average Recovery Time: {metrics['avg_recovery']} batches")
print(f"Median Recovery Time: {metrics['median_recovery']} batches")
print(f"Max Recovery Time: {metrics['max_recovery']} batches")
print(f"Estimated False Positive Rate: {metrics['false_positive_rate']}%")
print(f"Currently in Drift: {metrics['currently_in_drift']}")
print(f"\nRolling Accuracy Variance: {np.var(rolling_acc_history):.6f}")
print(f"Final Rolling Accuracy: {rolling_acc_history[-1]:.4f}")
print("=" * 60)

print(engine.get_drift_summary())

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Rolling Accuracy
axes[0].plot(rolling_acc_history, label='Rolling Accuracy', linewidth=1.5)
axes[0].axhline(y=np.mean(rolling_acc_history), color='gray', 
                linestyle='--', alpha=0.5, label='Mean')

# Mark drift points
for drift in engine.drift_log:
    color = 'red' if drift['type'] == 'Abrupt Drift' else 'orange'
    axes[0].axvline(x=drift['batch'], color=color, alpha=0.3, linestyle='--')
    axes[0].scatter(drift['batch'], rolling_acc_history[drift['batch']], 
                   color=color, s=100, zorder=5, marker='v')

axes[0].set_title("Hybrid Drift Adaptive Performance ", 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel("Batch")
axes[0].set_ylabel("Rolling Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Feature Shift
axes[1].plot(feature_shift_history, label='Feature Shift', 
             color='purple', alpha=0.7, linewidth=1)
axes[1].set_title("Feature Distribution Shift", fontsize=12)
axes[1].set_xlabel("Batch")
axes[1].set_ylabel("L2 Norm Shift")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Learning Rate
axes[2].plot(lr_history, label='Learning Rate', color='green', linewidth=1.5)
axes[2].set_title("Adaptive Learning Rate", fontsize=12)
axes[2].set_xlabel("Batch")
axes[2].set_ylabel("LR")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drift_analysis_tuned.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Visualization saved to drift_analysis_tuned.png")
print(" Execution completed successfully.")