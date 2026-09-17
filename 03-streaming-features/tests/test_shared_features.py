import pandas as pd
import pytest

from features.shared_features import rolling_vwap, rolling_volatility, momentum, compute_features


def test_rolling_vwap_hand_computed():
    prices = pd.Series([10.0, 20.0, 30.0])
    volumes = pd.Series([100, 100, 100])
    result = rolling_vwap(prices, volumes, window=3)
    # equal volumes -> vwap over full window == simple average at the end
    assert result.iloc[-1] == pytest.approx((10 + 20 + 30) / 3)


def test_rolling_vwap_weights_by_volume():
    prices = pd.Series([10.0, 20.0])
    volumes = pd.Series([100, 900])  # heavily weighted toward the second price
    result = rolling_vwap(prices, volumes, window=2)
    expected = (10 * 100 + 20 * 900) / (100 + 900)
    assert result.iloc[-1] == pytest.approx(expected)


def test_rolling_volatility_zero_for_constant_price():
    prices = pd.Series([100.0] * 10)
    result = rolling_volatility(prices, window=5)
    assert (result == 0).all()


def test_rolling_volatility_positive_for_varying_price():
    prices = pd.Series([100.0, 105.0, 98.0, 110.0, 95.0, 108.0])
    result = rolling_volatility(prices, window=3)
    assert result.iloc[-1] > 0


def test_momentum_hand_computed():
    prices = pd.Series([100.0, 102.0, 104.0, 106.0, 110.0])
    result = momentum(prices, window=2)
    # at index 4: (110 - 104) / 104
    assert result.iloc[4] == pytest.approx((110.0 - 104.0) / 104.0)


def test_momentum_early_values_are_zero_not_nan():
    prices = pd.Series([100.0, 102.0])
    result = momentum(prices, window=5)
    assert not result.isna().any()
    assert (result == 0).all()


def test_compute_features_adds_expected_columns():
    df = pd.DataFrame({
        "price": [100.0, 101.0, 102.0, 103.0, 104.0],
        "volume": [1000, 1000, 1000, 1000, 1000],
    })
    out = compute_features(df, vwap_window=3, vol_window=3, momentum_window=2)
    for col in ["vwap", "volatility", "momentum"]:
        assert col in out.columns
    assert len(out) == len(df)
