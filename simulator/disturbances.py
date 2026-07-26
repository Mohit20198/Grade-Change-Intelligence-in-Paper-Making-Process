"""
disturbances.py
---------------
Disturbance signal generators for the paper-making simulator.

Three disturbance types are supported:
  1. steam_sag    — step-down followed by slow recovery in steam pressure
  2. speed_hunting — sinusoidal oscillation in machine speed
  3. sensor_spike  — short burst of additive Gaussian noise on BW sensor

Each generator returns a delta value to ADD to the nominal signal at time t.
The caller is responsible for clipping to physical limits.
"""

import numpy as np
from typing import Optional


class SteamSagDisturbance:
    """
    Steam pressure sag: sudden drop then exponential recovery.

    Parameters
    ----------
    sag_magnitude   : peak pressure drop [bar], default 0.4
    recovery_tau    : recovery time constant [s], default 120
    duration        : total disturbance window [s], default 480
    """

    def __init__(self, sag_magnitude: float = 0.4,
                 recovery_tau: float = 120.0,
                 duration: int = 480):
        self.sag_magnitude = sag_magnitude
        self.recovery_tau = recovery_tau
        self.duration = duration

    def delta(self, dt_local: int) -> float:
        """dt_local = seconds since disturbance started (0-based)."""
        if dt_local < 0 or dt_local >= self.duration:
            return 0.0
        # Instant drop, exponential recovery toward 0
        return -self.sag_magnitude * np.exp(-dt_local / self.recovery_tau)


class SpeedHuntingDisturbance:
    """
    Machine speed hunting: sinusoidal oscillation.

    Parameters
    ----------
    amplitude   : peak speed deviation [m/min], default 8.0
    period      : oscillation period [s], default 60
    duration    : total disturbance window [s], default 300
    damping_tau : envelope decay [s], default 200
    """

    def __init__(self, amplitude: float = 8.0,
                 period: float = 60.0,
                 duration: int = 300,
                 damping_tau: float = 200.0):
        self.amplitude = amplitude
        self.period = period
        self.duration = duration
        self.damping_tau = damping_tau

    def delta(self, dt_local: int) -> float:
        if dt_local < 0 or dt_local >= self.duration:
            return 0.0
        envelope = np.exp(-dt_local / self.damping_tau)
        return self.amplitude * envelope * np.sin(2 * np.pi * dt_local / self.period)


class SensorSpikeDisturbance:
    """
    Sensor noise spike on basis weight scanner.
    Additive Gaussian burst for a short window.

    Parameters
    ----------
    spike_std   : std-dev of spike noise [gsm], default 4.0
    duration    : length of spike window [s], default 60
    rng_seed    : reproducibility
    """

    def __init__(self, spike_std: float = 4.0,
                 duration: int = 60,
                 rng_seed: Optional[int] = None):
        self.spike_std = spike_std
        self.duration = duration
        rng = np.random.default_rng(rng_seed)
        self._deltas = rng.normal(0, spike_std, size=duration + 10)

    def delta(self, dt_local: int) -> float:
        if dt_local < 0 or dt_local >= self.duration:
            return 0.0
        return float(self._deltas[dt_local])


class StockSurgeDisturbance:
    """
    Stock flow surge/sag: a sudden step in stock flow followed by recovery.
    This directly impacts Basis Weight (largest effect on quality).

    Positive surge → BW spike above setpoint.
    Negative surge (sag) → BW drops below setpoint.

    Parameters
    ----------
    magnitude       : flow deviation [L/min], can be +/-
    recovery_tau    : recovery time constant [s]
    duration        : total window [s]
    """

    def __init__(self, magnitude: float = 80.0,
                 recovery_tau: float = 90.0,
                 duration: int = 400):
        self.magnitude = magnitude
        self.recovery_tau = recovery_tau
        self.duration = duration

    def delta(self, dt_local: int) -> float:
        """Returns stock flow delta [L/min] at dt_local seconds since onset."""
        if dt_local < 0 or dt_local >= self.duration:
            return 0.0
        # Step onset then exponential recovery
        return self.magnitude * np.exp(-dt_local / self.recovery_tau)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_disturbance(dist_type: str, rng_seed: Optional[int] = None):
    """
    Instantiate a disturbance object by type name.

    dist_type : 'steam_sag' | 'speed_hunting' | 'sensor_spike' | 'stock_surge'
    """
    if dist_type == "steam_sag":
        return SteamSagDisturbance(
            sag_magnitude=np.random.uniform(0.25, 0.55),
            recovery_tau=np.random.uniform(90, 150),
            duration=int(np.random.uniform(360, 600)),
        )
    elif dist_type == "speed_hunting":
        return SpeedHuntingDisturbance(
            amplitude=np.random.uniform(30, 65),
            period=np.random.uniform(45, 90),
            duration=int(np.random.uniform(240, 420)),
            damping_tau=np.random.uniform(150, 300),
        )
    elif dist_type == "sensor_spike":
        return SensorSpikeDisturbance(
            spike_std=np.random.uniform(3.0, 6.0),
            duration=int(np.random.uniform(45, 90)),
            rng_seed=rng_seed,
        )
    elif dist_type == "stock_surge":
        # Randomly sag or surge, magnitude enough to push BW >2.5% off-spec
        sign = np.random.choice([-1, 1])
        magnitude = sign * np.random.uniform(60, 150)  # L/min
        return StockSurgeDisturbance(
            magnitude=magnitude,
            recovery_tau=np.random.uniform(60, 120),
            duration=int(np.random.uniform(300, 500)),
        )
    else:
        raise ValueError(f"Unknown disturbance type: {dist_type}")
