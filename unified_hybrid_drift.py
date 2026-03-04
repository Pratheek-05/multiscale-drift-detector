"""
============================================================
UNIFIED HYBRID DRIFT DETECTION SYSTEM
Combines Abrupt Concept Switching + Linear/Gradual Detection
Two drift types: ABRUPT and LINEAR/GRADUAL
============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import deque
import copy
import random

# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================

dataset = "gradual.csv"  # Change dataset here
batch_size = 50

# Learning rate settings
base_lr = 0.001
max_lr = 0.005

# Detection thresholds
abrupt_threshold = -0.10        # Sudden performance drop
gradual_slope_threshold = -0.002  # Gradual decline
feature_drift_threshold = 0.25   # Feature distribution change

# Windows and cooldown
performance_window = 15
rolling_window = 30
cooldown_period = 30

print(f"\n{'='*60}")
print(f"UNIFIED HYBRID DRIFT DETECTION SYSTEM")
print(f"{'='*60}")
print(f"Dataset: {dataset}")
print(f"Batch Size: {batch_size}")
print(f"Detection Mode: ABRUPT + LINEAR/GRADUAL")
print(f"{'='*60}\n")

# ============================================================
# MODEL
# ============================================================

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

# ============================================================
# UNIFIED DRIFT DETECTOR
# ============================================================

class UnifiedDriftDetector:
    def __init__(self, perf_window=15, roll_window=30, cooldown=30):
        self.perf_window = perf_window
        self.roll_window = roll_window
        self.cooldown = cooldown
        
        # Performance tracking
        self.accuracy_history = []
        self.performance_history = deque(maxlen=perf_window * 2)
        
        # Feature tracking
        self.feature_mean_ref = None
        self.feature_var_ref = None
        
        # Drift tracking
        self.drift_log = []
        self.cooldown_counter = 0
        self.last_drift_batch = -100
        
        # Concept memory for abrupt switching
        self.concept_memory = {}
        self.current_concept_id = 0
        self.concept_history = []
        
    def update_performance(self, accuracy):
        """Track performance for abrupt detection"""
        self.accuracy_history.append(accuracy)
        self.performance_history.append(accuracy)
        
        if len(self.performance_history) < self.perf_window * 2:
            return 0
        
        recent = np.mean(list(self.performance_history)[-self.perf_window:])
        previous = np.mean(list(self.performance_history)[-2*self.perf_window:-self.perf_window])
        
        return recent - previous
    
    def get_rolling_accuracy(self):
        """Get rolling accuracy for gradual detection"""
        if len(self.accuracy_history) >= self.roll_window:
            return np.mean(self.accuracy_history[-self.roll_window:])
        elif len(self.accuracy_history) > 0:
            return np.mean(self.accuracy_history)
        return 0.0
    
    def calculate_slope(self):
        """Calculate accuracy slope for gradual drift"""
        if len(self.accuracy_history) < self.roll_window:
            return 0.0
        
        y_vals = self.accuracy_history[-self.roll_window:]
        x_vals = np.arange(len(y_vals))
        slope = np.polyfit(x_vals, y_vals, 1)[0]
        
        return slope
    
    def check_feature_drift(self, X_batch):
        """Monitor feature distribution changes"""
        current_mean = np.mean(X_batch, axis=0)
        current_var = np.var(X_batch, axis=0)
        
        if self.feature_mean_ref is None:
            self.feature_mean_ref = current_mean
            self.feature_var_ref = current_var
            return 0.0, False
        
        mean_shift = np.linalg.norm(current_mean - self.feature_mean_ref)
        var_shift = np.linalg.norm(current_var - self.feature_var_ref)
        
        drift_score = mean_shift + var_shift
        drift_detected = drift_score > feature_drift_threshold
        
        return drift_score, drift_detected
    
    def detect(self, batch_idx, accuracy, X_batch, model):
        """
        Unified drift detection
        Returns: (drift_type, should_adapt, new_lr)
        drift_type: None, "ABRUPT", or "LINEAR/GRADUAL"
        """
        
        # Update cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.concept_history.append(self.current_concept_id)
            return None, False, base_lr
        
        # Get metrics
        perf_change = self.update_performance(accuracy)
        rolling_acc = self.get_rolling_accuracy()
        slope = self.calculate_slope()
        feature_score, feature_drift = self.check_feature_drift(X_batch)
        
        # ==================================================
        # ABRUPT DRIFT DETECTION (Priority 1)
        # ==================================================
        
        if perf_change < abrupt_threshold:
            # Strong negative performance change = ABRUPT
            
            drift_event = {
                'batch': batch_idx,
                'type': 'ABRUPT',
                'performance_change': perf_change,
                'rolling_acc': rolling_acc,
                'feature_drift': feature_score
            }
            
            self.drift_log.append(drift_event)
            
            print(f"\n🔴 ABRUPT DRIFT detected at batch {batch_idx}")
            print(f"   Performance drop: {perf_change:.4f}")
            print(f"   Rolling accuracy: {rolling_acc:.4f}")
            
            # Store current concept and switch
            self.concept_memory[self.current_concept_id] = copy.deepcopy(model.state_dict())
            self.current_concept_id = 1 - self.current_concept_id
            self.concept_history.append(self.current_concept_id)
            
            # Reset cooldown
            self.cooldown_counter = self.cooldown
            self.last_drift_batch = batch_idx
            
            # High learning rate for fast adaptation
            return "ABRUPT", True, 0.003
        
        # ==================================================
        # LINEAR/GRADUAL DRIFT DETECTION (Priority 2)
        # ==================================================
        
        # Need BOTH performance decline AND feature drift
        gradual_performance = slope < gradual_slope_threshold and abs(slope) > 0.001
        
        if gradual_performance and feature_drift:
            # Sustained decline + feature change = LINEAR/GRADUAL
            
            drift_event = {
                'batch': batch_idx,
                'type': 'LINEAR/GRADUAL',
                'slope': slope,
                'rolling_acc': rolling_acc,
                'feature_drift': feature_score
            }
            
            self.drift_log.append(drift_event)
            
            print(f"\n🟡 LINEAR/GRADUAL DRIFT detected at batch {batch_idx}")
            print(f"   Slope: {slope:.6f}")
            print(f"   Rolling accuracy: {rolling_acc:.4f}")
            print(f"   Feature drift: {feature_score:.4f}")
            
            self.concept_history.append(self.current_concept_id)
            
            # Reset cooldown
            self.cooldown_counter = self.cooldown
            self.last_drift_batch = batch_idx
            
            # Reset feature reference
            self.feature_mean_ref = np.mean(X_batch, axis=0)
            self.feature_var_ref = np.var(X_batch, axis=0)
            
            # Moderate learning rate increase
            drift_strength = abs(slope)
            adaptive_lr = base_lr * (1 + 5.0 * drift_strength)
            adaptive_lr = min(adaptive_lr, max_lr)
            
            return "LINEAR/GRADUAL", True, adaptive_lr
        
        # ==================================================
        # CONCEPT RECOVERY (for abrupt switching)
        # ==================================================
        
        # If performance improving strongly, might be returning to old concept
        if perf_change > 0.10:
            if self.current_concept_id in self.concept_memory:
                
                print(f"\n🟢 Potential concept recovery at batch {batch_idx}")
                print(f"   Performance improvement: {perf_change:.4f}")
                
                # Load previous concept
                model.load_state_dict(self.concept_memory[self.current_concept_id])
                
                self.concept_history.append(self.current_concept_id)
                self.cooldown_counter = self.cooldown
                
                return "RECOVERY", True, base_lr
        
        # No drift detected
        self.concept_history.append(self.current_concept_id)
        return None, False, base_lr
    
    def get_summary(self):
        """Get detection summary"""
        if not self.drift_log:
            return {
                'total_drifts': 0,
                'abrupt_count': 0,
                'gradual_count': 0
            }
        
        abrupt_count = sum(1 for d in self.drift_log if d['type'] == 'ABRUPT')
        gradual_count = sum(1 for d in self.drift_log if d['type'] == 'LINEAR/GRADUAL')
        
        return {
            'total_drifts': len(self.drift_log),
            'abrupt_count': abrupt_count,
            'gradual_count': gradual_count,
            'drift_log': self.drift_log
        }

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(dataset)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.astype(np.int64)  # Ensure integer labels

scaler = StandardScaler()
X = scaler.fit_transform(X)

total_batches = len(X) // batch_size

print(f"Total Samples: {len(X)}")
print(f"Total Batches: {total_batches}")
print(f"Features: {X.shape[1]}")
print(f"\n{'='*60}\n")

# ============================================================
# INITIALIZE SYSTEM
# ============================================================

model = SimpleClassifier(X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=base_lr)
criterion = nn.CrossEntropyLoss()

detector = UnifiedDriftDetector(
    perf_window=performance_window,
    roll_window=rolling_window,
    cooldown=cooldown_period
)

# Tracking
lr_history = []
drift_markers = []  # For plotting

# ============================================================
# STREAMING TRAINING LOOP
# ============================================================

print("Starting training...")
print(f"{'='*60}\n")

for batch_idx in range(total_batches):
    
    start = batch_idx * batch_size
    end = start + batch_size
    
    X_batch = X[start:end]
    y_batch = y[start:end]
    
    X_tensor = torch.tensor(X_batch, dtype=torch.float32)
    y_tensor = torch.tensor(y_batch, dtype=torch.long)
    
    # ========================================
    # EVALUATE BEFORE TRAINING
    # ========================================
    
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == y_tensor).float().mean().item()
    
    # ========================================
    # DRIFT DETECTION
    # ========================================
    
    drift_type, should_adapt, new_lr = detector.detect(
        batch_idx, accuracy, X_batch, model
    )
    
    if should_adapt:
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Mark for plotting
        drift_markers.append((batch_idx, drift_type))
    
    current_lr = optimizer.param_groups[0]['lr']
    lr_history.append(current_lr)
    
    # ========================================
    # TRAIN MODEL
    # ========================================
    
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Progress update
    if batch_idx % 50 == 0:
        rolling_acc = detector.get_rolling_accuracy()
        print(f"Batch {batch_idx:4d}/{total_batches} | "
              f"Acc: {accuracy:.4f} | "
              f"Rolling: {rolling_acc:.4f} | "
              f"LR: {current_lr:.6f}")

# ============================================================
# FINAL SUMMARY
# ============================================================

summary = detector.get_summary()

print(f"\n{'='*60}")
print("DRIFT DETECTION SUMMARY")
print(f"{'='*60}")
print(f"Total Drifts Detected: {summary['total_drifts']}")
print(f"  🔴 Abrupt: {summary['abrupt_count']}")
print(f"  🟡 Linear/Gradual: {summary['gradual_count']}")
print(f"{'='*60}\n")

if summary['drift_log']:
    print("Drift Events:")
    for i, drift in enumerate(summary['drift_log'], 1):
        print(f"  {i}. Batch {drift['batch']:4d}: {drift['type']}")

print(f"\n{'='*60}")
print("Execution completed successfully!")
print(f"{'='*60}\n")

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Accuracy
axes[0, 0].plot(detector.accuracy_history, label='Batch Accuracy', alpha=0.6)
rolling_acc_full = [detector.get_rolling_accuracy() if i >= rolling_window 
                    else np.mean(detector.accuracy_history[:i+1]) 
                    for i in range(len(detector.accuracy_history))]
axes[0, 0].plot(rolling_acc_full, label='Rolling Accuracy', linewidth=2)

# Mark drifts
for batch, drift_type in drift_markers:
    color = 'red' if drift_type == 'ABRUPT' else 'orange'
    if batch < len(detector.accuracy_history):
        axes[0, 0].axvline(x=batch, color=color, alpha=0.3, linestyle='--')
        axes[0, 0].scatter(batch, detector.accuracy_history[batch], 
                          color=color, s=100, marker='v', zorder=5)

axes[0, 0].set_xlabel('Batch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy with Drift Detection')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Learning Rate
axes[0, 1].plot(lr_history, color='green', linewidth=2)
axes[0, 1].set_xlabel('Batch')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Adaptive Learning Rate')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Concept History
axes[1, 0].plot(detector.concept_history, color='purple', linewidth=2)
axes[1, 0].set_xlabel('Batch')
axes[1, 0].set_ylabel('Concept ID')
axes[1, 0].set_title('Concept Switching (Abrupt Detection)')
axes[1, 0].set_yticks([0, 1])
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Drift Type Distribution
drift_types = [d['type'] for d in summary['drift_log']]
if drift_types:
    from collections import Counter
    counts = Counter(drift_types)
    axes[1, 1].bar(counts.keys(), counts.values(), 
                   color=['red' if k == 'ABRUPT' else 'orange' for k in counts.keys()])
    axes[1, 1].set_xlabel('Drift Type')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Drift Type Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
else:
    axes[1, 1].text(0.5, 0.5, 'No Drifts Detected', 
                    ha='center', va='center', fontsize=14)
    axes[1, 1].set_title('Drift Type Distribution')

plt.tight_layout()
plt.savefig('hybrid_drift_detection.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'hybrid_drift_detection.png'")
