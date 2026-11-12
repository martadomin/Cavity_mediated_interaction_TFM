"""
Secondary (top) x-axis in pump units for figures plotted against V0 (Kac-scaled, units of eps).

V0 is proportional to eta^2 (pump amplitude squared), and the GP threshold for g = 0, N -> infinity is
|V0| = 2 pi^2 eps = hbar omega_R. Hence, without any cavity parameters,
    P / P_cr     = eta^2 / eta_c^2 = |V0| / (2 pi^2 eps)      (pump power, as on the axes of Baumann, Mottl)
    eta / eta_c  = sqrt(|V0| / (2 pi^2 eps))                   (pump amplitude)
The finite-N threshold 2 pi^2 N/(N-1) sits at P/P_cr = N/(N-1) (1.026 at N = 40).

Usage:  from pump_axis import add_pump_axis;  add_pump_axis(ax)            # P/P_cr (default)
                                               add_pump_axis(ax, "eta")     # eta/eta_c
"""

import numpy as np

EPS_R = 2 * np.pi ** 2          # hbar omega_R in units of eps


def add_pump_axis(ax, kind="P"):
    if kind == "eta":
        fwd = lambda v: np.sqrt(np.abs(v) / EPS_R)
        inv = lambda e: -EPS_R * np.asarray(e) ** 2
        label = r"$\eta/\eta_c$"
    else:
        fwd = lambda v: np.abs(v) / EPS_R
        inv = lambda p: -EPS_R * np.asarray(p)
        label = r"$P/P_{\rm cr}$"
    sec = ax.secondary_xaxis("top", functions=(fwd, inv))
    if kind == "eta":
        sec.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])          # square-root map: automatic ticks crowd near 0
    sec.set_xlabel(label)
    return sec
