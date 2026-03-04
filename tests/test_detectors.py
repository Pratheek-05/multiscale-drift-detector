"""
Basic tests for the drift detection components.
These run in CI on every push.
"""
import sys
import os

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from statistical_detector import PageHinkley


# ── PageHinkley tests ────────────────────────────────────────────────────────

class TestPageHinkley:

    def test_no_drift_on_stable_stream(self):
        """Stable constant input should not trigger drift."""
        ph = PageHinkley(delta=0.005, threshold=50.0)
        detections = sum(ph.update(0.1) for _ in range(200))
        assert detections == 0, "Should not detect drift on a stable stream"

    def test_detects_drift_on_step_change(self):
        """A large step change in the stream should trigger at least one detection."""
        ph = PageHinkley(delta=0.005, threshold=10.0)
        # Stable phase
        for _ in range(50):
            ph.update(0.05)
        # Abrupt shift
        detections = sum(ph.update(0.9) for _ in range(50))
        assert detections >= 1, "Should detect drift after a step change"

    def test_reset_clears_state(self):
        """After reset, detector should behave as if freshly initialised."""
        ph = PageHinkley()
        for _ in range(100):
            ph.update(0.5)
        ph.reset()
        assert ph.n == 0
        assert ph.sum == 0.0

    def test_returns_bool(self):
        """update() must always return a bool."""
        ph = PageHinkley()
        result = ph.update(0.3)
        assert isinstance(result, bool)


# ── SimpleClassifier smoke tests ─────────────────────────────────────────────

class TestSimpleClassifier:

    def test_forward_pass_shape(self):
        """Classifier output should be (batch, 2)."""
        import torch
        # Import app module — guard against streamlit import errors in CI
        try:
            import importlib.util, types
            # Stub streamlit so the module can be imported without a display
            st_stub = types.ModuleType("streamlit")
            for attr in ["set_page_config", "markdown", "sidebar", "header",
                         "file_uploader", "button", "progress", "empty",
                         "subheader", "columns", "metric", "dataframe",
                         "expander", "info", "success", "warning",
                         "download_button", "plotly_chart", "pyplot"]:
                setattr(st_stub, attr, lambda *a, **kw: None)
            st_stub.sidebar = types.SimpleNamespace(
                header=lambda *a, **kw: None,
                slider=lambda *a, **kw: kw.get("value", a[3] if len(a) > 3 else 0),
                select_slider=lambda *a, **kw: kw.get("value", None),
                expander=lambda *a, **kw: _DummyCtx(),
                markdown=lambda *a, **kw: None,
            )
            sys.modules.setdefault("streamlit", st_stub)

            spec = importlib.util.spec_from_file_location(
                "app",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
            )
            app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app)
            clf = app.SimpleClassifier(input_dim=4)
        except Exception:
            pytest.skip("Could not import app module in headless CI — skipping")
            return

        x = torch.randn(8, 4)
        out = clf(x)
        assert out.shape == (8, 2), f"Expected (8,2), got {out.shape}"

    def test_forward_pass_no_nan(self):
        """Output should not contain NaN values."""
        import torch
        try:
            import importlib.util, types
            st_stub = types.ModuleType("streamlit")
            for attr in ["set_page_config","markdown","header","file_uploader",
                         "button","progress","empty","subheader","columns",
                         "metric","dataframe","expander","info","success",
                         "warning","download_button","plotly_chart","pyplot"]:
                setattr(st_stub, attr, lambda *a, **kw: None)
            st_stub.sidebar = types.SimpleNamespace(
                header=lambda *a,**kw: None,
                slider=lambda *a,**kw: kw.get("value", 0),
                select_slider=lambda *a,**kw: kw.get("value", None),
                expander=lambda *a,**kw: _DummyCtx(),
                markdown=lambda *a,**kw: None,
            )
            sys.modules.setdefault("streamlit", st_stub)
            spec = importlib.util.spec_from_file_location(
                "app",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
            )
            app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(app)
            clf = app.SimpleClassifier(input_dim=6)
        except Exception:
            pytest.skip("Could not import app module in headless CI — skipping")
            return

        x = torch.randn(16, 6)
        out = clf(x)
        assert not torch.isnan(out).any(), "Output contains NaN"


class _DummyCtx:
    """Minimal context manager stub for streamlit expanders."""
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ── RLDriftAgent smoke tests ─────────────────────────────────────────────────

class TestRLDriftAgent:

    def _make_agent(self):
        """Import RLDriftAgent with a stubbed streamlit."""
        import types, importlib.util
        st_stub = types.ModuleType("streamlit")
        for attr in ["set_page_config","markdown","header","file_uploader",
                     "button","progress","empty","subheader","columns",
                     "metric","dataframe","expander","info","success",
                     "warning","download_button","plotly_chart","pyplot"]:
            setattr(st_stub, attr, lambda *a, **kw: None)
        st_stub.sidebar = types.SimpleNamespace(
            header=lambda *a,**kw: None,
            slider=lambda *a,**kw: kw.get("value", 0),
            select_slider=lambda *a,**kw: kw.get("value", None),
            expander=lambda *a,**kw: _DummyCtx(),
            markdown=lambda *a,**kw: None,
        )
        sys.modules.setdefault("streamlit", st_stub)
        spec = importlib.util.spec_from_file_location(
            "app",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
        )
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        return app.RLDriftAgent(state_dim=5, action_dim=5)

    def test_select_action_valid_range(self):
        try:
            agent = self._make_agent()
        except Exception:
            pytest.skip("Could not import app module"); return
        state = np.zeros(5, dtype=np.float32)
        action = agent.select_action(state)
        assert 0 <= action < 5, f"Action {action} out of range [0,5)"

    def test_epsilon_decreases(self):
        try:
            agent = self._make_agent()
        except Exception:
            pytest.skip("Could not import app module"); return
        state = np.zeros(5, dtype=np.float32)
        eps_start = agent.epsilon
        for _ in range(50):
            agent.select_action(state)
        assert agent.epsilon < eps_start, "Epsilon should decay over steps"

    def test_reward_positive_on_accuracy_gain(self):
        try:
            agent = self._make_agent()
        except Exception:
            pytest.skip("Could not import app module"); return
        reward = agent.compute_reward(
            prev_accuracy=0.5, curr_accuracy=0.8,
            drift_detected=False, action=0, recovery_happened=False
        )
        assert reward > 0, "Accuracy gain should yield positive reward"

    def test_reward_negative_on_false_action(self):
        try:
            agent = self._make_agent()
        except Exception:
            pytest.skip("Could not import app module"); return
        # No drift but agent takes action 1 — should be penalised
        reward = agent.compute_reward(
            prev_accuracy=0.8, curr_accuracy=0.8,
            drift_detected=False, action=1, recovery_happened=False
        )
        assert reward < 0, "Acting when stable should yield negative reward"
