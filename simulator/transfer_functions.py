"""
transfer_functions.py
---------------------
First-order lag + deadtime (FOLPD) transfer function implementations
for the paper-making process simulator.

Each function maintains a state buffer and advances one timestep (dt).
Usage: instantiate TransferFunction objects and call .step(u) repeatedly.

All time constants and deadtimes are in seconds.
"""

import numpy as np
from collections import deque


class TransferFunction:
    """
    First-Order-Lag + Deadtime (FOLPD) transfer function.

    G(s) = K * exp(-theta*s) / (tau*s + 1)

    Discretized via Euler integration:
        y[k] = y[k-1] + (dt/tau) * (K * u_delayed[k] - y[k-1])

    Parameters
    ----------
    gain        : process gain K
    tau         : time constant [s]
    deadtime    : deadtime theta [s]
    dt          : sample time [s], default 1
    initial_out : initial output value (steady-state)
    initial_in  : initial input value (steady-state)
    """

    def __init__(self, gain: float, tau: float, deadtime: float,
                 dt: float = 1.0, initial_out: float = 0.0,
                 initial_in: float = 0.0):
        self.gain = gain
        self.tau = tau
        self.deadtime = deadtime
        self.dt = dt

        # Deadtime buffer: store enough samples to cover theta
        n_delay = max(1, int(round(deadtime / dt)))
        self.delay_buffer = deque([initial_in] * n_delay, maxlen=n_delay)

        # Current output (integrator state)
        self.y = initial_out

    def step(self, u: float) -> float:
        """Advance one timestep given input u, return current output."""
        # Push new input, pop oldest (that becomes the delayed signal)
        self.delay_buffer.append(u)
        u_delayed = self.delay_buffer[0]

        # First-order lag update
        self.y = self.y + (self.dt / self.tau) * (self.gain * u_delayed - self.y)
        return self.y

    def reset(self, output: float, input_val: float):
        """Re-initialize state (e.g. after a setpoint jump)."""
        self.y = output
        self.delay_buffer = deque(
            [input_val] * self.delay_buffer.maxlen,
            maxlen=self.delay_buffer.maxlen
        )


# ---------------------------------------------------------------------------
# Named factory functions — tune gains / lags here in one place
# ---------------------------------------------------------------------------

# Reference operating point (G60 grade) used for gain calibration
# All absolute-input TFs are calibrated so that at G60 steady state
# their outputs exactly equal the G60 CV target.
REF_STOCK_FLOW    = 1000.0   # L/min
REF_MACHINE_SPEED = 880.0    # m/min
REF_STEAM_PRESSURE = 4.5     # bar
REF_FILLER_FLOW   = 175.0    # kg/min
REF_BW            = 60.0     # gsm
REF_MOISTURE      = 6.0      # %
REF_ASH           = 12.0     # %


def make_stock_to_bw(dt: float = 1.0, initial_out: float = 60.0,
                     initial_in: float = 1000.0) -> TransferFunction:
    """
    Stock flow [L/min] → Basis Weight [gsm]
    Gain calibrated: K = REF_BW / REF_STOCK_FLOW = 60/1000 = 0.06 gsm/(L/min)
    This gives BW = 60 gsm when flow = 1000 L/min (G60 steady state).
    Tau:  45 s  (sheet formation + scanner transport)
    Deadtime: 15 s (headbox to scanner)
    Tunable: change gain, tau, deadtime here.
    """
    gain = REF_BW / REF_STOCK_FLOW   # = 0.06
    return TransferFunction(
        gain=gain,
        tau=45.0,
        deadtime=15.0,
        dt=dt,
        initial_out=initial_out,
        initial_in=initial_in,
    )


def make_speed_to_bw(dt: float = 1.0, initial_out: float = 0.0,
                     initial_in: float = 0.0) -> TransferFunction:
    """
    Machine speed DEVIATION from nominal [m/min] → BW delta [gsm]
    Inverse: speed up → weight drops (fibre dilution at headbox).
    Gain: -0.068 gsm per m/min (calibrated: REF_BW / REF_SPEED = 60/880 ≈ 0.068)
    Sign: NEGATIVE because faster speed dilutes the sheet.
    Tau:  30 s
    Deadtime: 10 s
    Input is (machine_speed - REF_MACHINE_SPEED); output is BW delta.
    """
    gain = -(REF_BW / REF_MACHINE_SPEED)   # ≈ -0.068
    return TransferFunction(
        gain=gain,
        tau=30.0,
        deadtime=10.0,
        dt=dt,
        initial_out=initial_out,
        initial_in=initial_in,
    )


def make_steam_to_moisture(dt: float = 1.0, initial_out: float = 6.0,
                            initial_in: float = 4.5) -> TransferFunction:
    """
    Steam pressure [bar] → Moisture [%]
    Inverse: more steam dries the sheet → moisture decreases.
    Calibration: at G60, moisture_target=6.0% at steam=4.5 bar.
    Gain = -REF_MOISTURE / REF_STEAM = -6.0/4.5 = -1.333 %/bar
    (negative because higher pressure → lower moisture)
    Tau:  150 s (thermal mass of dryer section)
    Deadtime: 30 s
    """
    gain = -REF_MOISTURE / REF_STEAM_PRESSURE   # ≈ -1.333
    return TransferFunction(
        gain=gain,
        tau=150.0,
        deadtime=30.0,
        dt=dt,
        initial_out=initial_out,
        initial_in=initial_in,
    )


def make_speed_to_moisture(dt: float = 1.0, initial_out: float = 0.0,
                            initial_in: float = 0.0) -> TransferFunction:
    """
    Machine speed DEVIATION [m/min] → Moisture delta [%]
    HIDDEN / UNDOCUMENTED relationship (from paper machine patents):
    Higher speed → less dwell time in dryer → sheet exits wetter.
    Gain: +0.003 %/(m/min deviation)
    Tau:  90 s  (long lag — this IS the 'secret' correlation to discover)
    Deadtime: 45 s
    Input is (machine_speed - REF_MACHINE_SPEED).
    """
    return TransferFunction(
        gain=0.003,
        tau=90.0,
        deadtime=45.0,
        dt=dt,
        initial_out=initial_out,
        initial_in=initial_in,
    )


def make_filler_to_ash(dt: float = 1.0, initial_out: float = 12.0,
                        initial_in: float = 200.0) -> TransferFunction:
    """
    Filler flow [kg/min] → Ash content [%]
    Near-direct coupling.
    Gain: +0.06 %/(kg/min)
    Tau:  5-10 s
    Deadtime: 3 s
    """
    return TransferFunction(
        gain=0.06,
        tau=7.0,
        deadtime=3.0,
        dt=dt,
        initial_out=initial_out,
        initial_in=initial_in,
    )
