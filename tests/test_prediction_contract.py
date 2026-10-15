"""Point-in-time and executable-outcome regression tests."""
import numpy as np
import pandas as pd
import pytest

import features as F
import labeling as L
import validation as V
from tests.test_features import _candles


def test_model_inputs_ignore_unavailable_venue_extras():
    candles = _candles()
    basic = candles.drop(columns="taker_buy_volume")
    with_flow = F.compute_model_features(candles)
    without_flow = F.compute_model_features(basic)
    # v4 contract: taker_buy_volume flows through when the venue provides it;
    # every v2/v3 feature must be identical either way, and nothing beyond the
    # documented flow columns may appear.
    extra = set(with_flow.columns) - set(without_flow.columns)
    assert extra == set(F.V4_COLUMNS) or extra == set()
    pd.testing.assert_frame_equal(with_flow.drop(columns=sorted(extra), errors="ignore"),
                                  without_flow.drop(columns=sorted(extra), errors="ignore"))


def test_missing_features_are_not_fabricated():
    with pytest.raises(ValueError, match="Missing model"):
        F.prepare_model_input(pd.DataFrame({"ret": [1.]}), ["tbr_5m"])
    with pytest.raises(ValueError, match="Nonfinite"):
        F.prepare_model_input(pd.DataFrame({"ret": [np.nan]}), ["ret"])


def test_gap_invalidates_lookbacks():
    candles = _candles(4000).drop(index=2000)
    feats = F.compute_model_features(candles)
    assert feats.iloc[2000:3441].isna().all().all()
    assert feats.iloc[-1].notna().all()


def test_future_changes_cannot_change_past_features():
    candles = _candles(2500)
    expected = F.compute_model_features(candles)
    candles.loc[2200:, ["open", "high", "low", "close", "volume"]] *= 2
    actual = F.compute_model_features(candles)
    pd.testing.assert_frame_equal(expected.iloc[:2200], actual.iloc[:2200])


def test_stale_btc_is_not_forward_filled():
    sol, btc = _candles(), _candles(start=50000.)
    f = F.compute_model_features(sol, btc.iloc[:-10])
    assert f["btc_ret_1m"].iloc[-10:].isna().all()


def test_internal_roles_are_purged_and_disjoint():
    idx = pd.date_range("2025-01-01", periods=4000, freq="min", tz="UTC")
    parts = V.temporal_partitions(idx, 60)
    arrays = list(parts.values())
    for a, b in zip(arrays[:-1], arrays[1:]):
        assert idx[a[-1]] + pd.Timedelta(minutes=61) < idx[b[0]]
        assert not set(a) & set(b)


def _path(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"],
                        index=pd.date_range("2025-01-01", periods=len(rows),
                                            freq="min", tz="UTC"))


def test_next_open_not_signal_close_and_real_time_exit():
    c = _path([[100, 100, 100, 100], [105, 108, 104, 107],
               [107, 111, 106, 110], [110, 111, 109, 110]])
    r = V.path_returns(c, [1, 1, 1, 0], 2, cost=.001)
    assert r[0] == pytest.approx(110 / 105 - 1 - .001)
    assert np.isnan(r[1])  # first position remains open
    assert np.isnan(r[-1])


def test_stop_first_when_both_barriers_touched():
    c = _path([[100, 100, 100, 100], [100, 104, 98, 102], [102, 103, 101, 102]])
    r = V.path_returns(c, [1, 0, 0], 1, np.full(3, .01), cost=0)
    assert r[0] == pytest.approx(-.01)


def test_gap_through_stop_uses_worse_open():
    c = _path([[100, 100, 100, 100], [100, 100.5, 99.5, 100],
               [95, 96, 94, 95]])
    assert V.path_returns(c, [1, 0, 0], 2, np.full(3, .01), cost=0)[0] == pytest.approx(-.05)


def test_forward_target_is_symmetric_and_tail_unknown():
    c = pd.DataFrame({"close": [100, 102, 100, 98, 100]})
    result = L.forward_return_labels(c, 1, .01)
    assert result.iloc[:4].tolist() == [1, -1, -1, 1]
    assert np.isnan(result.iloc[-1])


def test_barrier_target_does_not_cross_missing_minute():
    c = _path([[100, 100, 100, 100]] * 8).drop(
        pd.Timestamp("2025-01-01 00:02", tz="UTC"))
    labels = L.triple_barrier_labels(c, pd.Series(1., index=c.index), 3)
    assert np.isnan(labels.iloc[0])


def test_probability_metrics_penalize_overconfidence():
    y = [0, 1, 2]
    uniform = V.probability_metrics(y, np.full((3, 3), 1 / 3))
    good = V.probability_metrics(y, np.eye(3) * .8 + .2 / 3)
    assert good["brier"] < uniform["brier"]
    assert good["log_loss"] < uniform["log_loss"]
    assert good["accuracy"] == 1.


def test_threshold_replay_does_not_let_rejected_signals_block_trades():
    import train
    c = _path([[100, 100, 100, 100]] * 5)
    p = np.array([[.1, .5, .4], [.05, .9, .05], [1, 0, 0],
                  [1, 0, 0], [1, 0, 0]])
    result = train._threshold_diagnostics(
        [1] * 5, p, c, c.index, np.full(5, .01), 2, [.8])
    assert result[0]["signals"] == 1


def test_path_retains_future_bars_outside_prediction_slice():
    import train
    c = _path([[100, 100, 100, 100], [100, 100.5, 99.5, 100],
               [100, 100.5, 99.5, 100]])
    result = train._path(c, c.index[:1], [1], [.01], 2)
    assert result[0] == pytest.approx(-.0013)


def test_drawdown_includes_initial_capital_and_compounds():
    assert V.max_drawdown([-.1]) == pytest.approx(.1)
    assert V.max_drawdown([-.1, -.1]) == pytest.approx(.19)