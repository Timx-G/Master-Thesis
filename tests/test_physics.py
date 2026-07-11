import math

import pytest

from uav_mec.model import ChannelConfig, NOMAChannel, noise_power_dbm, noise_power_w


def test_noise_for_one_subcarrier_includes_nf_once():
    bandwidth_hz = 5e6 / 3
    assert noise_power_dbm(bandwidth_hz, -174.0, 7.0) == pytest.approx(-104.7815, abs=1e-3)
    noise_dbm = 10 * math.log10(noise_power_w(bandwidth_hz, -174.0, 7.0)) + 30
    assert noise_dbm == pytest.approx(-104.7815, abs=1e-3)


def test_channel_uses_urban_parameters_and_db_averaged_loss():
    cfg = ChannelConfig()
    parameters = (cfg.environment, cfg.a_los, cfg.b_los, cfg.eta_los, cfg.eta_nlos)
    assert parameters == ("urban", 9.61, 0.16, 1.0, 20.0)
    assert not hasattr(cfg, "noise_power_dbm")
    channel = NOMAChannel(cfg, 100.0)
    p_los = channel._los_probability(400.0)
    distance = math.hypot(400.0, 100.0)
    fspl_db = 20 * math.log10(4 * math.pi * cfg.fc * distance / 3e8)
    expected_gain = 10 ** (-(fspl_db + p_los * 1.0 + (1 - p_los) * 20.0) / 10)
    assert channel.path_loss(400.0) == pytest.approx(expected_gain, rel=1e-12)
