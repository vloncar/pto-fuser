"""M3 off-NPU tests — graph-replay backend structure (no device required).

The capture/replay behaviour itself needs an NPU (see test_m3_npu.py); here we
cover the host-side contract: the capture-mode toggle that lets opaque lowerings
drop their stream syncs, and the guard that replay cannot run before capture.
"""
from pto_fuser import GraphReplayExecutor, capture_mode
from pto_fuser import registry as reg


def test_capture_mode_toggles_and_restores():
    assert reg._CAPTURE_MODE is False
    with capture_mode():
        assert reg._CAPTURE_MODE is True
    assert reg._CAPTURE_MODE is False


def test_capture_mode_restores_on_exception():
    try:
        with capture_mode():
            assert reg._CAPTURE_MODE is True
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert reg._CAPTURE_MODE is False


def test_capture_mode_nests():
    with capture_mode():
        with capture_mode():
            assert reg._CAPTURE_MODE is True
        # inner exit must not clear the flag while the outer region is active
        assert reg._CAPTURE_MODE is True
    assert reg._CAPTURE_MODE is False


def test_replay_before_capture_raises():
    gr = GraphReplayExecutor()
    try:
        gr.replay({})
    except RuntimeError as e:
        assert "capture()" in str(e)
    else:
        raise AssertionError("replay before capture should raise RuntimeError")
