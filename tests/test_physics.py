import math

import numpy as np
import pytest

from uav_mec.model import ChannelConfig, NOMAChannel, noise_power_dbm, noise_power_w


def test_noise_for_one_subcarrier_includes_nf_once():
    bandwidth_hz = 5e6 / 3
    assert noise_power_dbm(bandwidth_hz, -174.0, 7.0) == pytest.approx(-104.7815, abs=1e-3)
    noise_dbm = 10 * math.log10(noise_power_w(bandwidth_hz, -174.0, 7.0)) + 30
    assert noise_dbm == pytest.approx(-104.7815, abs=1e-3)


def test_full_and_subcarrier_noise_share_one_density():
    channel = NOMAChannel(ChannelConfig(), 100.0)

    assert channel.noise_linear == pytest.approx(9.976311574844412e-14, rel=1e-12)
    assert channel.noise_for_bandwidth(5e6 / 3) == pytest.approx(
        3.325437191614803e-14,
        rel=1e-12,
    )


def test_legacy_full_band_noise_override_updates_subcarrier_noise():
    cfg = ChannelConfig()
    channel = NOMAChannel(cfg, 100.0)

    channel.noise_linear = 6e-12

    expected_psd_dbm_hz = (
        10.0 * math.log10(6e-12) + 30.0 - cfg.noise_figure_db - 10.0 * math.log10(cfg.bandwidth)
    )
    assert cfg.noise_psd_dbm_hz == pytest.approx(expected_psd_dbm_hz, rel=1e-12)
    assert channel.noise_linear == pytest.approx(6e-12, rel=1e-12, abs=0.0)
    assert channel.noise_for_bandwidth(channel.cfg.bandwidth / 3) == pytest.approx(
        2e-12,
        rel=1e-12,
        abs=0.0,
    )


def test_channel_noise_tracks_mutable_config():
    cfg = ChannelConfig()
    channel = NOMAChannel(cfg, 100.0)

    cfg.noise_figure_db = 9.0
    assert channel.noise_linear == pytest.approx(noise_power_w(5e6, -174.0, 9.0))

    cfg.noise_psd_dbm_hz = -171.0
    assert channel.noise_linear == pytest.approx(noise_power_w(5e6, -171.0, 9.0))

    cfg.bandwidth = 2e6
    assert channel.noise_linear == pytest.approx(noise_power_w(2e6, -171.0, 9.0))

    cfg.noise_figure_db = 4.0

    assert channel.noise_linear == pytest.approx(noise_power_w(2e6, -171.0, 4.0))
    assert channel.noise_for_bandwidth(5e5) == pytest.approx(noise_power_w(5e5, -171.0, 4.0))


@pytest.mark.parametrize("bandwidth_hz", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_noise_helpers_reject_invalid_bandwidth(bandwidth_hz):
    with pytest.raises(ValueError, match="bandwidth_hz must be finite and positive"):
        noise_power_dbm(bandwidth_hz)


@pytest.mark.parametrize(
    ("noise_psd_dbm_hz", "noise_figure_db"),
    [(math.nan, 7.0), (math.inf, 7.0), (-174.0, math.nan), (-174.0, math.inf)],
)
def test_noise_helpers_reject_non_finite_receiver_parameters(
    noise_psd_dbm_hz,
    noise_figure_db,
):
    with pytest.raises(ValueError, match="must be finite"):
        noise_power_dbm(5e6, noise_psd_dbm_hz, noise_figure_db)


def test_channel_config_accepts_new_positional_field_order():
    cfg = ChannelConfig(4e6, "urban", -173.0, 6.0, 9.0, 0.2, 2.4e9, 2.0, 18.0, 0.1)

    assert cfg == ChannelConfig(
        bandwidth=4e6,
        environment="urban",
        noise_psd_dbm_hz=-173.0,
        noise_figure_db=6.0,
        a_los=9.0,
        b_los=0.2,
        fc=2.4e9,
        eta_los=2.0,
        eta_nlos=18.0,
        p_max=0.1,
    )


def test_channel_config_remains_mutable():
    cfg = ChannelConfig()

    cfg.bandwidth = 1e6

    assert cfg.bandwidth == 1e6


def test_legacy_noise_positional_shape_fails_explicitly():
    with pytest.raises(ValueError, match="environment must be a non-empty string"):
        ChannelConfig(5e6, -104.8, 7.0)


def test_channel_allows_finite_receiver_gain():
    cfg = ChannelConfig(noise_figure_db=-1.0)

    assert NOMAChannel(cfg, 100.0).noise_linear == pytest.approx(
        noise_power_w(cfg.bandwidth, cfg.noise_psd_dbm_hz, -1.0)
    )


@pytest.mark.parametrize("uav_height", [0.0, -1.0, math.nan, math.inf])
def test_channel_rejects_invalid_uav_height(uav_height):
    with pytest.raises(ValueError, match="uav_height must be finite and positive"):
        NOMAChannel(ChannelConfig(), uav_height)


@pytest.mark.parametrize("distance", [-1.0, math.nan, math.inf, -math.inf])
def test_path_loss_rejects_invalid_horizontal_distance(distance):
    channel = NOMAChannel(ChannelConfig(), 100.0)

    with pytest.raises(ValueError, match="d_horizontal must be finite and non-negative"):
        channel.path_loss(distance)


@pytest.mark.parametrize("fc", [0.0, -1.0, math.nan, math.inf])
def test_channel_rejects_invalid_carrier_frequency(fc):
    cfg = ChannelConfig()
    cfg.fc = fc

    with pytest.raises(ValueError, match="fc must be finite and positive"):
        NOMAChannel(cfg, 100.0)


def test_path_loss_revalidates_mutable_carrier_frequency():
    cfg = ChannelConfig()
    channel = NOMAChannel(cfg, 100.0)
    cfg.fc = 0.0

    with pytest.raises(ValueError, match="fc must be finite and positive"):
        channel.path_loss(400.0)


def test_channel_uses_urban_parameters_and_db_averaged_loss():
    cfg = ChannelConfig()
    parameters = (cfg.environment, cfg.a_los, cfg.b_los, cfg.eta_los, cfg.eta_nlos)
    assert parameters == ("urban", 9.61, 0.16, 1.0, 20.0)
    assert not hasattr(cfg, "noise_power_dbm")
    channel = NOMAChannel(cfg, 100.0)

    assert channel._los_probability(400.0) == pytest.approx(0.17442211932670978, rel=1e-12)
    assert channel.path_loss(400.0) == pytest.approx(1.7976945235825014e-11, rel=1e-12)


def test_3gpp_uma_av_audited_pathloss_matches_tr_36777_annex_b_at_100m():
    cfg = ChannelConfig(pathloss_model="3gpp_uma_av_audited")
    channel = NOMAChannel(cfg, 100.0)
    horizontal_m = 400.0
    distance_3d_m = math.hypot(horizontal_m, 100.0)
    p1_m = 4300.0 * math.log10(100.0) - 3800.0
    d1_m = max(460.0 * math.log10(100.0) - 700.0, 18.0)
    p_los = d1_m / horizontal_m + math.exp(-horizontal_m / p1_m) * (
        1.0 - d1_m / horizontal_m
    )
    los_db = 28.0 + 22.0 * math.log10(distance_3d_m) + 20.0 * math.log10(2.0)
    nlos_db = (
        -17.5
        + (46.0 - 7.0 * math.log10(100.0)) * math.log10(distance_3d_m)
        + 20.0 * math.log10(40.0 * math.pi * 2.0 / 3.0)
    )
    expected_gain = 10.0 ** (-(p_los * los_db + (1.0 - p_los) * nlos_db) / 10.0)

    assert channel.path_loss(horizontal_m) == pytest.approx(expected_gain, rel=1e-12)


@pytest.mark.parametrize("model", ("", "3gpp", "invented"))
def test_channel_config_rejects_unregistered_pathloss_model(model):
    with pytest.raises(ValueError, match="pathloss_model"):
        ChannelConfig(pathloss_model=model)


def test_ofdma_uses_noise_for_each_users_bandwidth():
    channel = NOMAChannel(ChannelConfig(), 100.0)
    uav_pos = np.array([0.0, 0.0, 100.0])
    user_positions = np.array([[0.0, 0.0], [0.0, 0.0]])
    powers = np.array([0.2, 0.2])
    bw_per_user = channel.cfg.bandwidth / 2
    expected = bw_per_user * math.log2(
        1.0 + powers[0] * channel.path_loss(0.0) / channel.noise_for_bandwidth(bw_per_user)
    )

    rates = channel.compute_ofdma_rates(uav_pos, user_positions, powers)

    assert rates == pytest.approx([expected, expected], rel=1e-12)


def test_multi_uav_ofdma_uses_noise_for_each_users_bandwidth():
    channel = NOMAChannel(ChannelConfig(), 100.0)
    uav_positions = [np.array([0.0, 0.0, 100.0]), np.array([100.0, 0.0, 100.0])]
    user_positions = np.array([[0.0, 0.0], [100.0, 0.0]])
    powers = np.array([0.2, 0.2])
    assignment = np.array([0, 1])
    bw_per_user = channel.cfg.bandwidth / 2
    expected = bw_per_user * math.log2(
        1.0 + powers[0] * channel.path_loss(0.0) / channel.noise_for_bandwidth(bw_per_user)
    )

    rates = channel.compute_multi_uav_ofdma_rates(
        uav_positions,
        user_positions,
        powers,
        assignment,
    )

    assert rates == pytest.approx([expected, expected], rel=1e-12)
