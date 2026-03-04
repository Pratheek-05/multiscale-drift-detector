"""
UNIFIED HYBRID DRIFT DETECTION DASHBOARD
+ Hybrid vs Statistical (Page-Hinkley) — run together batch-by-batch
+ Performance stats noted for both methods
+ Drift point comparison table
+ Drift-ONLY zoomed live preview (no full graph shown during run)
+ Accuracy dip graph with animated drift markers
+ RL AGENT (DQN) for adaptive drift response
+ Simple UI — one button, results appear below
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from collections import deque
import copy
import time
import random

from statistical_detector import PageHinkley

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hybrid Drift Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-header{
  font-size:2.4rem;font-weight:bold;
  background:linear-gradient(90deg,#FF6B6B 0%,#FFA500 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  text-align:center;padding:0.8rem 0;
}
.card{padding:12px;border-radius:10px;margin:6px 0;}
.drift-abrupt{background:#ff6b6b;color:white;font-weight:bold;}
.drift-gradual{background:#ffa500;color:white;font-weight:bold;}
.status-green {background:#1db954;color:white;font-weight:bold;text-align:center;padding:14px;border-radius:10px;}
.status-orange{background:#ff9f1a;color:white;font-weight:bold;text-align:center;padding:14px;border-radius:10px;}
.status-red   {background:#e63946;color:white;font-weight:bold;text-align:center;padding:14px;border-radius:10px;}
.alert-box{
  border-left:5px solid #e63946;
  background:#fff0f0;color:#333;
  padding:10px 14px;border-radius:6px;margin:4px 0;
  animation: fadeIn 0.5s ease-in;
}
.alert-box-grad{
  border-left:5px solid #ffa500;
  background:#fff8e7;color:#333;
  padding:10px 14px;border-radius:6px;margin:4px 0;
  animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(-6px);}to{opacity:1;transform:translateY(0);}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CLASSIFIER MODEL
# ============================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# RL AGENT (DQN)
# ============================================================

class DQNNetwork(nn.Module):
    def __init__(self, state_dim=5, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),        nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states,      dtype=np.float32),
                np.array(actions),
                np.array(rewards,     dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones,       dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class RLDriftAgent:
    def __init__(self, state_dim=5, action_dim=5,
                 lr=1e-3, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=300,
                 batch_size=32, target_update=20):

        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps_done    = 0

        self.policy_net = DQNNetwork(state_dim, action_dim)
        self.target_net = DQNNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer()

        self.rl_action_log  = []
        self.rl_reward_log  = []
        self.rl_loss_log    = []
        self.rl_epsilon_log = []

        self.ACTION_NAMES = {
            0: "HOLD", 1: "BOOST_LR", 2: "ADAPTIVE_LR",
            3: "RESTORE_CONCEPT", 4: "RESET_FEATURE_REF"
        }

    def _encode_state(self, perf_change, rolling_acc, slope, feature_score, cooldown_active):
        return np.array([
            np.clip(perf_change,   -1.0, 1.0),
            np.clip(rolling_acc,    0.0, 1.0),
            np.clip(slope,         -0.1, 0.1),
            np.clip(feature_score,  0.0, 5.0) / 5.0,
            float(cooldown_active)
        ], dtype=np.float32)

    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * \
                       np.exp(-self.steps_done / self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.policy_net(torch.tensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def compute_reward(self, prev_accuracy, curr_accuracy, drift_detected, action, recovery_happened):
        reward = (curr_accuracy - prev_accuracy) * 10.0
        if drift_detected and action in (1, 2):
            reward += 0.5
        if recovery_happened:
            reward += 1.0
        if action != 0 and not drift_detected:
            reward -= 0.2
        return float(reward)

    def store_transition(self, state, action, reward, next_state, done=False):
        self.replay.push(state, action, reward, next_state, done)
        self.rl_reward_log.append(reward)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states_t      = torch.tensor(states)
        actions_t     = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t     = torch.tensor(rewards)
        next_states_t = torch.tensor(next_states)
        dones_t       = torch.tensor(dones)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0]
            target_q   = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.rl_loss_log.append(loss.item())

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def log_action(self, batch_idx, action):
        self.rl_action_log.append((batch_idx, self.ACTION_NAMES[action]))
        self.rl_epsilon_log.append(self.epsilon)

# ============================================================
# UNIFIED DRIFT DETECTOR (RL-driven)
# ============================================================

class UnifiedDriftDetector:
    def __init__(self, perf_window, roll_window, cooldown,
                 abrupt_thresh, gradual_thresh, feature_thresh,
                 rl_agent: RLDriftAgent):
        self.perf_window       = perf_window
        self.roll_window       = roll_window
        self.cooldown          = cooldown
        self.abrupt_threshold  = abrupt_thresh
        self.gradual_threshold = gradual_thresh
        self.feature_threshold = feature_thresh

        self.accuracy_history    = []
        self.performance_history = deque(maxlen=perf_window * 2)
        self.feature_mean_ref    = None
        self.feature_var_ref     = None
        self.drift_log           = []
        self.cooldown_counter    = 0
        self.concept_memory      = {}
        self.current_concept_id  = 0
        self.concept_history     = []

        self.rl_agent       = rl_agent
        self._prev_accuracy = 0.5

    def update_performance(self, accuracy):
        self.accuracy_history.append(accuracy)
        self.performance_history.append(accuracy)
        if len(self.performance_history) < self.perf_window * 2:
            return 0
        recent   = np.mean(list(self.performance_history)[-self.perf_window:])
        previous = np.mean(list(self.performance_history)[-2*self.perf_window:-self.perf_window])
        return recent - previous

    def get_rolling_accuracy(self):
        if len(self.accuracy_history) >= self.roll_window:
            return np.mean(self.accuracy_history[-self.roll_window:])
        elif self.accuracy_history:
            return np.mean(self.accuracy_history)
        return 0.0

    def calculate_slope(self):
        if len(self.accuracy_history) < self.roll_window:
            return 0.0
        y_vals = self.accuracy_history[-self.roll_window:]
        x_vals = np.arange(len(y_vals))
        return np.polyfit(x_vals, y_vals, 1)[0]

    def check_feature_drift(self, X_batch):
        current_mean = np.mean(X_batch, axis=0)
        current_var  = np.var(X_batch, axis=0)
        if self.feature_mean_ref is None:
            self.feature_mean_ref = current_mean
            self.feature_var_ref  = current_var
            return 0.0, False
        mean_shift = np.linalg.norm(current_mean - self.feature_mean_ref)
        var_shift  = np.linalg.norm(current_var  - self.feature_var_ref)
        score = mean_shift + var_shift
        return score, score > self.feature_threshold

    def detect(self, batch_idx, accuracy, X_batch, model, base_lr, max_lr):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.concept_history.append(self.current_concept_id)
            perf_change      = self.update_performance(accuracy)
            rolling_acc      = self.get_rolling_accuracy()
            slope            = self.calculate_slope()
            feature_score, _ = self.check_feature_drift(X_batch)
            state  = self.rl_agent._encode_state(perf_change, rolling_acc, slope, feature_score, True)
            reward = self.rl_agent.compute_reward(self._prev_accuracy, accuracy, False, 0, False)
            self.rl_agent.store_transition(state, 0, reward, state)
            self.rl_agent.learn()
            self.rl_agent.log_action(batch_idx, 0)
            self._prev_accuracy = accuracy
            return None, False, base_lr

        perf_change              = self.update_performance(accuracy)
        rolling_acc              = self.get_rolling_accuracy()
        slope                    = self.calculate_slope()
        feature_score, feature_drift = self.check_feature_drift(X_batch)

        state  = self.rl_agent._encode_state(perf_change, rolling_acc, slope, feature_score,
                                              cooldown_active=(self.cooldown_counter > 0))
        action = self.rl_agent.select_action(state)

        drift_type        = None
        should_adapt      = False
        new_lr            = base_lr
        recovery_happened = False

        abrupt_signal  = perf_change < self.abrupt_threshold
        gradual_signal = (slope < self.gradual_threshold and abs(slope) > 0.001 and feature_drift)

        if action == 1:
            ev = {'batch': batch_idx, 'type': 'ABRUPT',
                  'performance_change': perf_change, 'rolling_acc': rolling_acc,
                  'feature_drift': feature_score}
            self.drift_log.append(ev)
            self.concept_memory[self.current_concept_id] = copy.deepcopy(model.state_dict())
            self.current_concept_id = 1 - self.current_concept_id
            self.cooldown_counter   = self.cooldown
            drift_type   = "ABRUPT"
            should_adapt = True
            new_lr       = 0.003

        elif action == 2:
            ev = {'batch': batch_idx, 'type': 'LINEAR',
                  'slope': slope, 'rolling_acc': rolling_acc, 'feature_drift': feature_score}
            self.drift_log.append(ev)
            self.cooldown_counter = self.cooldown
            self.feature_mean_ref = np.mean(X_batch, axis=0)
            self.feature_var_ref  = np.var(X_batch, axis=0)
            adaptive_lr  = min(base_lr * (1 + 5.0 * abs(slope)), max_lr)
            drift_type   = "LINEAR"
            should_adapt = True
            new_lr       = adaptive_lr

        elif action == 3:
            if self.current_concept_id in self.concept_memory:
                model.load_state_dict(self.concept_memory[self.current_concept_id])
                recovery_happened = True
                self.cooldown_counter = self.cooldown
                should_adapt = True
                drift_type   = "RECOVERY"
                new_lr       = base_lr

        elif action == 4:
            self.feature_mean_ref = np.mean(X_batch, axis=0)
            self.feature_var_ref  = np.var(X_batch, axis=0)

        next_rolling_acc      = self.get_rolling_accuracy()
        next_slope            = self.calculate_slope()
        next_feature_score, _ = self.check_feature_drift(X_batch)
        next_state = self.rl_agent._encode_state(
            perf_change, next_rolling_acc, next_slope, next_feature_score,
            cooldown_active=(self.cooldown_counter > 0))

        drift_detected = abrupt_signal or gradual_signal
        reward = self.rl_agent.compute_reward(
            self._prev_accuracy, accuracy, drift_detected, action, recovery_happened)

        self.rl_agent.store_transition(state, action, reward, next_state)
        self.rl_agent.learn()
        self.rl_agent.log_action(batch_idx, action)

        self._prev_accuracy = accuracy
        self.concept_history.append(self.current_concept_id)
        return drift_type, should_adapt, new_lr

# ============================================================
# TRAINING — both models run batch-by-batch
# ============================================================

def train_and_detect(X, y, batch_size, config, progress_bar, status_text, live_placeholder):
    if y.dtype in (np.float64, np.float32):
        y = y.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    total_batches = len(X) // batch_size

    model     = SimpleClassifier(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=config['base_lr'])
    criterion = nn.CrossEntropyLoss()

    rl_agent = RLDriftAgent(
        state_dim=5, action_dim=5,
        lr=config.get('rl_lr', 1e-3),
        gamma=config.get('rl_gamma', 0.95),
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=config.get('rl_epsilon_decay', 300),
        batch_size=32, target_update=20
    )

    detector = UnifiedDriftDetector(
        perf_window=config['perf_window'],
        roll_window=config['roll_window'],
        cooldown=config['cooldown'],
        abrupt_thresh=config['abrupt_threshold'],
        gradual_thresh=config['gradual_threshold'],
        feature_thresh=config['feature_threshold'],
        rl_agent=rl_agent
    )

    stat_detector      = PageHinkley()
    lr_history         = []
    drift_markers      = []
    stat_drift_points  = []

    for batch_idx in range(total_batches):
        s = batch_idx * batch_size
        e = s + batch_size
        X_batch = X[s:e]
        y_batch = y[s:e]

        X_tensor = torch.tensor(X_batch, dtype=torch.float32)
        y_tensor = torch.tensor(y_batch, dtype=torch.long)

        with torch.no_grad():
            outputs  = model(X_tensor)
            preds    = torch.argmax(outputs, dim=1)
            accuracy = (preds == y_tensor).float().mean().item()

        # Statistical model
        if stat_detector.update(1 - accuracy):
            stat_drift_points.append(batch_idx)

        # Hybrid (RL-driven)
        drift_type, should_adapt, new_lr = detector.detect(
            batch_idx, accuracy, X_batch, model,
            config['base_lr'], config['max_lr']
        )
        if should_adapt:
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            drift_markers.append((batch_idx, drift_type))

        lr_history.append(optimizer.param_groups[0]['lr'])

        # Train
        outputs = model(X_tensor)
        loss    = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Status update
        if batch_idx % 20 == 0:
            progress_bar.progress((batch_idx + 1) / total_batches)
            rolling = detector.get_rolling_accuracy()
            status_text.markdown(
                f"**Batch {batch_idx}/{total_batches}** &nbsp;|&nbsp; "
                f"Hybrid Acc: `{accuracy:.3f}` &nbsp;|&nbsp; "
                f"Rolling: `{rolling:.3f}` &nbsp;|&nbsp; "
                f"Hybrid drifts: **{len(drift_markers)}** &nbsp;|&nbsp; "
                f"Statistical drifts: **{len(stat_drift_points)}**"
            )

        # Live drift-zone preview — only show around latest drift (mentor #5)
        if batch_idx % 40 == 0 and batch_idx > 0 and drift_markers:
            latest_drift = drift_markers[-1][0]
            w  = 60
            ws = max(0, latest_drift - w)
            we = min(len(detector.accuracy_history), latest_drift + w)
            if we > ws:
                zoom_acc = detector.accuracy_history[ws:we]
                zoom_lr  = lr_history[ws:we]
                zoom_dm  = [(b - ws, t) for b, t in drift_markers if ws <= b < we]
                zoom_ch  = detector.concept_history[ws:we]
                live_fig = _drift_zone_plot(
                    zoom_acc, zoom_lr, zoom_dm, zoom_ch,
                    title=f"Live Drift Zone Preview (batches {ws}–{we})")
                live_placeholder.plotly_chart(live_fig, use_container_width=True)

    progress_bar.progress(1.0)
    status_text.markdown("**Training complete!**")

    return {
        'detector':          detector,
        'lr_history':        lr_history,
        'drift_markers':     drift_markers,
        'stat_drift_points': stat_drift_points,
        'rl_agent':          rl_agent,
        'total_batches':     total_batches
    }

# ============================================================
# PLOT HELPERS
# ============================================================

def _drift_zone_plot(accuracy_history, lr_history, drift_markers, concept_history, title="Drift Zone"):
    """Compact zoomed plot — only the drift window, not the full graph."""
    fig = make_subplots(rows=2, cols=1,
        subplot_titles=('Accuracy (Drift Zone)', 'Learning Rate'),
        vertical_spacing=0.18, row_heights=[0.65, 0.35])
    batches = list(range(len(accuracy_history)))

    fig.add_trace(go.Scatter(x=batches, y=accuracy_history, mode='lines',
        name='Accuracy', line=dict(color='#1f77b4', width=2)), row=1, col=1)

    # Group markers by type — one legend entry per type, no inline text labels
    abrupt_x, abrupt_y, gradual_x, gradual_y = [], [], [], []
    for b, t in drift_markers:
        if b < len(accuracy_history):
            if t == 'ABRUPT':
                abrupt_x.append(b); abrupt_y.append(accuracy_history[b])
            else:
                gradual_x.append(b); gradual_y.append(accuracy_history[b])

    if abrupt_x:
        fig.add_trace(go.Scatter(
            x=abrupt_x, y=abrupt_y, mode='markers', name='Abrupt Drift',
            marker=dict(color='#e63946', size=12, symbol='triangle-down',
                        line=dict(color='white', width=1))
        ), row=1, col=1)
    if gradual_x:
        fig.add_trace(go.Scatter(
            x=gradual_x, y=gradual_y, mode='markers', name='Gradual Drift',
            marker=dict(color='#ffa500', size=12, symbol='triangle-down',
                        line=dict(color='white', width=1))
        ), row=1, col=1)

    fig.add_trace(go.Scatter(x=batches, y=lr_history, mode='lines',
        name='LR', line=dict(color='#2ca02c', width=2), showlegend=False), row=2, col=1)

    fig.update_layout(
        height=380, title_text=title,
        legend=dict(
            orientation='h', yanchor='top', y=-0.15,
            xanchor='center', x=0.5, font=dict(size=12)
        ),
        margin=dict(t=60, b=60, l=60, r=20)
    )
    fig.update_xaxes(title_text="Relative Batch", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="LR", row=2, col=1)
    return fig


def _accuracy_dip_plot(accuracy_history, drift_markers, stat_drift_points):
    """
    Accuracy dip graph — both models' drift points overlaid.
    Legend is deduplicated: one entry per marker type, placed below chart.
    """
    fig = go.Figure()

    # Accuracy line
    fig.add_trace(go.Scatter(
        x=list(range(len(accuracy_history))), y=accuracy_history,
        mode='lines', name='Accuracy',
        line=dict(color='#1f77b4', width=2)
    ))

    # Collect coords by type — ONE legend entry per type
    abrupt_x, abrupt_y   = [], []
    gradual_x, gradual_y = [], []
    recov_x,  recov_y    = [], []

    for b, t in drift_markers:
        if b < len(accuracy_history):
            if t == 'ABRUPT':
                abrupt_x.append(b); abrupt_y.append(accuracy_history[b])
            elif t == 'LINEAR':
                gradual_x.append(b); gradual_y.append(accuracy_history[b])
            else:  # RECOVERY
                recov_x.append(b); recov_y.append(accuracy_history[b])

    stat_x, stat_y = [], []
    for b in stat_drift_points:
        if b < len(accuracy_history):
            stat_x.append(b); stat_y.append(accuracy_history[b])

    if abrupt_x:
        fig.add_trace(go.Scatter(
            x=abrupt_x, y=abrupt_y, mode='markers', name='Hybrid — Abrupt',
            marker=dict(color='#e63946', size=12, symbol='triangle-down',
                        line=dict(color='white', width=1))
        ))
    if gradual_x:
        fig.add_trace(go.Scatter(
            x=gradual_x, y=gradual_y, mode='markers', name='Hybrid — Gradual',
            marker=dict(color='#ffa500', size=12, symbol='triangle-down',
                        line=dict(color='white', width=1))
        ))
    if recov_x:
        fig.add_trace(go.Scatter(
            x=recov_x, y=recov_y, mode='markers', name='Hybrid — Recovery',
            marker=dict(color='#2ecc71', size=12, symbol='triangle-up',
                        line=dict(color='white', width=1))
        ))
    if stat_x:
        fig.add_trace(go.Scatter(
            x=stat_x, y=stat_y, mode='markers', name='Statistical (PH)',
            marker=dict(color='#9b59b6', size=11, symbol='x-thin',
                        line=dict(color='#9b59b6', width=2))
        ))

    # Subtle vertical lines — only for abrupt drifts to avoid clutter
    for b in abrupt_x:
        fig.add_vline(x=b, line_dash="dot", line_color="rgba(230,57,70,0.25)", line_width=1)

    fig.update_layout(
        title=dict(text="Accuracy Dip — Hybrid & Statistical", x=0.5, xanchor='center'),
        xaxis_title="Batch",
        yaxis_title="Accuracy",
        height=420,
        margin=dict(t=60, b=100, l=60, r=20),
        legend=dict(
            orientation='h',
            yanchor='top', y=-0.22,
            xanchor='center', x=0.5,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#ddd', borderwidth=1
        )
    )
    return fig


def _comparison_bar_chart(hybrid_drifts, stat_drifts, hybrid_abrupt, hybrid_gradual):
    """Side-by-side drift count comparison (mentor #3 & #4)."""
    fig = go.Figure(data=[
        go.Bar(name='Hybrid Model',
               x=['Total Drifts', 'Abrupt', 'Gradual'],
               y=[hybrid_drifts, hybrid_abrupt, hybrid_gradual],
               marker_color=['#1f77b4', '#e63946', '#ffa500']),
        go.Bar(name='Statistical (PH)',
               x=['Total Drifts', 'Abrupt', 'Gradual'],
               y=[stat_drifts, stat_drifts, 0],
               marker_color=['#9b59b6', '#9b59b6', '#cccccc'])
    ])
    fig.update_layout(
        barmode='group',
        title=dict(text='Drift Detection Comparison — Both Models', x=0.5, xanchor='center'),
        height=340,
        margin=dict(t=55, b=80, l=50, r=20),
        legend=dict(
            orientation='h', yanchor='top', y=-0.22,
            xanchor='center', x=0.5, font=dict(size=12)
        )
    )
    return fig


def _rl_diagnostics_plot(rl_agent: RLDriftAgent):
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=('Cumulative Reward', 'DQN Training Loss',
                        'Epsilon Decay', 'Action Distribution'),
        vertical_spacing=0.28, horizontal_spacing=0.15)

    if rl_agent.rl_reward_log:
        cum = np.cumsum(rl_agent.rl_reward_log)
        fig.add_trace(go.Scatter(x=list(range(len(cum))), y=cum, mode='lines',
            name='Cumul. Reward', showlegend=False,
            line=dict(color='#2ecc71', width=2)), row=1, col=1)

    if rl_agent.rl_loss_log:
        fig.add_trace(go.Scatter(x=list(range(len(rl_agent.rl_loss_log))),
            y=rl_agent.rl_loss_log, mode='lines',
            name='DQN Loss', showlegend=False,
            line=dict(color='#e74c3c', width=1.5)), row=1, col=2)

    if rl_agent.rl_epsilon_log:
        fig.add_trace(go.Scatter(x=list(range(len(rl_agent.rl_epsilon_log))),
            y=rl_agent.rl_epsilon_log, mode='lines',
            name='Epsilon', showlegend=False,
            line=dict(color='#9b59b6', width=2)), row=2, col=1)

    if rl_agent.rl_action_log:
        action_counts = {n: 0 for n in rl_agent.ACTION_NAMES.values()}
        for _, a in rl_agent.rl_action_log:
            action_counts[a] += 1
        fig.add_trace(go.Bar(
            x=list(action_counts.keys()), y=list(action_counts.values()),
            showlegend=False,
            marker_color=['#3498db','#e74c3c','#f39c12','#2ecc71','#9b59b6']
        ), row=2, col=2)

    fig.update_layout(
        height=560, showlegend=False,
        title=dict(text="RL Agent Diagnostics", x=0.5, xanchor='center'),
        margin=dict(t=70, b=40, l=60, r=40)
    )
    # Axis labels
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_xaxes(title_text="Action", row=2, col=2, tickangle=-20)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="Loss",   row=1, col=2)
    fig.update_yaxes(title_text="ε",      row=2, col=1)
    fig.update_yaxes(title_text="Count",  row=2, col=2)
    return fig

# ============================================================
# MAIN
# ============================================================

def main():
    st.markdown('<p class="main-header">Hybrid Drift Detection Dashboard</p>', unsafe_allow_html=True)

    # Compact sidebar — all settings collapsed by default (mentor: fewer clicks)
    with st.sidebar:
        st.header(" Settings")
        batch_size = st.slider("Batch Size", 25, 100, 50, 25)

        with st.expander("Detection Parameters", expanded=False):
            abrupt_threshold  = st.slider("Abrupt Threshold", -0.20, -0.05, -0.10, 0.01)
            gradual_threshold = st.slider("Gradual Slope Threshold", -0.010, -0.001, -0.002, 0.001, format="%.4f")
            feature_threshold = st.slider("Feature Drift Threshold", 0.10, 0.50, 0.25, 0.05)

        with st.expander("Advanced Settings", expanded=False):
            perf_window = st.slider("Performance Window", 10, 30, 15, 5)
            roll_window = st.slider("Rolling Window", 20, 50, 30, 5)
            cooldown    = st.slider("Cooldown Period", 20, 50, 30, 5)
            base_lr     = st.select_slider("Base LR", [0.0001,0.0002,0.0005,0.001,0.002], value=0.001)
            max_lr      = st.select_slider("Max LR",  [0.002,0.003,0.005,0.008,0.01],     value=0.005)

        with st.expander("RL Agent Settings", expanded=False):
            rl_lr            = st.select_slider("RL LR", [0.0001,0.0005,0.001,0.005,0.01], value=0.001)
            rl_gamma         = st.slider("RL Gamma", 0.80, 0.99, 0.95, 0.01)
            rl_epsilon_decay = st.slider("Epsilon Decay Steps", 100, 1000, 300, 50)

    # Upload
    uploaded_file = st.file_uploader(
        " Upload CSV dataset  (last column = label 0/1)",
        type=['csv']
    )

    if uploaded_file is None:
        st.info("Upload a CSV to get started. Adjust settings via the sidebar if needed.")
        return

    df = pd.read_csv(uploaded_file)

    # Compact dataset overview
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples",  df.shape[0])
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Classes",  df.iloc[:, -1].nunique())
    with st.expander("Preview dataset", expanded=False):
        st.dataframe(df.head(8), use_container_width=True)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Single button (mentor: fewer clicks)
    if not st.button(" Run Drift Detection", type="primary", use_container_width=True):
        return

    st.markdown("---")

    config = dict(
        base_lr=base_lr, max_lr=max_lr,
        perf_window=perf_window, roll_window=roll_window, cooldown=cooldown,
        abrupt_threshold=abrupt_threshold, gradual_threshold=gradual_threshold,
        feature_threshold=feature_threshold,
        rl_lr=rl_lr, rl_gamma=rl_gamma, rl_epsilon_decay=rl_epsilon_decay
    )

    progress_bar     = st.progress(0)
    status_text      = st.empty()
    live_placeholder = st.empty()   # drift-zone only, live

    results = train_and_detect(X, y, batch_size, config,
                               progress_bar, status_text, live_placeholder)
    live_placeholder.empty()

    detector      = results['detector']
    rl_agent      = results['rl_agent']
    drift_log     = detector.drift_log
    drift_markers = results['drift_markers']
    stat_points   = results['stat_drift_points']

    abrupt_count  = sum(1 for d in drift_log if d['type'] == 'ABRUPT')
    gradual_count = sum(1 for d in drift_log if d['type'] == 'LINEAR')

    # ── Status banner ──────────────────────────────────────────────────────
    if not drift_log:
        st.markdown('<div class="status-green">✅ SYSTEM STATUS: STABLE — No drift detected</div>',
                    unsafe_allow_html=True)
    elif drift_log[-1]['type'] == 'ABRUPT':
        st.markdown('<div class="status-red">🔴 SYSTEM STATUS: CRITICAL — Abrupt drift detected</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-orange">🟠 SYSTEM STATUS: GRADUAL DRIFT detected</div>',
                    unsafe_allow_html=True)

    st.markdown("")

    # ── Summary metrics — both models (mentor #1, #2, #3) ─────────────────
    st.subheader(" Results — Both Models")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Hybrid Total Drifts",  len(drift_log))
    m2.metric("Hybrid Abrupt",        abrupt_count)
    m3.metric("Hybrid Gradual",       gradual_count)
    m4.metric("Statistical (PH)",     len(stat_points))
    m5.metric("Final Rolling Acc",    f"{detector.get_rolling_accuracy():.3f}")

    st.markdown("")

    # Comparison bar chart (mentor #3 & #4)
    st.plotly_chart(
        _comparison_bar_chart(len(drift_log), len(stat_points), abrupt_count, gradual_count),
        use_container_width=True
    )

    # Drift batch comparison table (mentor #4)
    with st.expander(" Drift Batch Comparison Table", expanded=True):
        comp_len = max(len(drift_markers), len(stat_points), 1)
        hyb_b  = [b for b, _ in drift_markers] + [None] * (comp_len - len(drift_markers))
        hyb_t  = [t for _, t in drift_markers] + [None] * (comp_len - len(drift_markers))
        stat_b = stat_points + [None] * (comp_len - len(stat_points))
        st.dataframe(pd.DataFrame({
            "Hybrid Batch":           hyb_b,
            "Hybrid Type":            hyb_t,
            "Statistical Batch (PH)": stat_b
        }), use_container_width=True)

    st.markdown("---")

    # ── Drift Zone Views — one expander per drift event (mentor #5) ────────
    st.subheader("Drift Zone Views")
    if drift_markers:
        for i, (focus, dtype) in enumerate(drift_markers):
            w  = 60
            ws = max(0, focus - w)
            we = min(len(detector.accuracy_history), focus + w)
            if we > ws:
                zoom_acc = detector.accuracy_history[ws:we]
                zoom_lr  = results['lr_history'][ws:we]
                zoom_dm  = [(b - ws, t) for b, t in drift_markers if ws <= b < we]
                zoom_ch  = detector.concept_history[ws:we]
                label    = "🔴 Abrupt" if dtype == "ABRUPT" else "🟠 Gradual"
                with st.expander(f"Drift #{i+1} — {label} at batch {focus}", expanded=(i == 0)):
                    st.plotly_chart(
                        _drift_zone_plot(zoom_acc, zoom_lr, zoom_dm, zoom_ch,
                                         title=f"Drift #{i+1} Zone (batches {ws}–{we})"),
                        use_container_width=True
                    )
    else:
        st.info("No drift detected — no zones to show.")

    st.markdown("---")

    # ── Accuracy dip graph (mentor #6) ────────────────────────────────────
    st.subheader(" Accuracy Dip Graph")
    st.plotly_chart(
        _accuracy_dip_plot(detector.accuracy_history, drift_markers, stat_points),
        use_container_width=True
    )

    st.markdown("---")

    # ── Animated drift alerts (mentor later #1) ────────────────────────────
    st.subheader(" Drift Alerts")
    if drift_markers:
        for b, t in drift_markers:
            css_class = "alert-box" if t == "ABRUPT" else "alert-box-grad"
            icon      = "🔴" if t == "ABRUPT" else "🟠"
            st.markdown(
                f'<div class="{css_class}">'
                f'{icon} <strong>{t} drift</strong> — batch <strong>{b}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.15)
    else:
        st.success("No drift alerts — model remained stable.")

    st.markdown("---")

    # ── RL Agent Diagnostics ───────────────────────────────────────────────
    st.subheader(" RL Agent Diagnostics")
    st.plotly_chart(_rl_diagnostics_plot(rl_agent), use_container_width=True)

    with st.expander("RL Action Log (last 50 steps)"):
        if rl_agent.rl_action_log:
            st.dataframe(pd.DataFrame(rl_agent.rl_action_log[-50:],
                                      columns=["Batch", "Action"]),
                         use_container_width=True)

    if rl_agent.rl_reward_log:
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Total RL Reward", f"{sum(rl_agent.rl_reward_log):.2f}")
        rc2.metric("Avg RL Reward",   f"{np.mean(rl_agent.rl_reward_log):.3f}")
        rc3.metric("Final Epsilon",    f"{rl_agent.epsilon:.3f}")

    st.markdown("---")

    # ── Download report ────────────────────────────────────────────────────
    report = pd.DataFrame([{
        "Hybrid_Total":           len(drift_log),
        "Hybrid_Abrupt":          abrupt_count,
        "Hybrid_Gradual":         gradual_count,
        "Statistical_Total":      len(stat_points),
        "Final_Rolling_Accuracy": detector.get_rolling_accuracy(),
        "RL_Total_Reward":        sum(rl_agent.rl_reward_log) if rl_agent.rl_reward_log else 0,
        "RL_Avg_Reward":          np.mean(rl_agent.rl_reward_log) if rl_agent.rl_reward_log else 0,
        "RL_Final_Epsilon":       rl_agent.epsilon
    }])
    st.download_button("⬇️ Download Experiment Report", report.to_csv(index=False),
                       "drift_report.csv", "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()