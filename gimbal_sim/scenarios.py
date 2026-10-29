"""Движение основания (внешнее воздействие). Начальная ошибка угла сюда НЕ входит: это начальное состояние."""
from __future__ import annotations

import numpy as np


def smootherstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def base_signal(p, n: int, h: float):
    """Угол основания, скорость и ускорение на сетке из n+1 точек с шагом h (град, град/с, град/с^2)."""
    t = np.arange(n + 1) * h
    A = p.dist_amp_deg
    t0 = p.dist_t0_s
    tr = p.dist_trans_ms * 1e-3
    sc = p.scenario
    if sc == "initial":
        th = np.zeros_like(t)
    elif sc == "step":
        th = A * smootherstep((t - t0) / tr)
    elif sc == "pulse":
        th = A * (smootherstep((t - t0) / tr) - smootherstep((t - t0 - p.dist_dur_s) / tr))
    elif sc == "sine":
        th = A * np.sin(2.0 * np.pi * p.dist_freq_hz * (t - t0)) * (t >= t0)
    elif sc == "random":
        rng = np.random.default_rng(p.noise_seed + 12345)
        w = rng.normal(size=t.size)
        a = 1.0 - np.exp(-2.0 * np.pi * p.dist_rand_fc_hz * h)
        x = np.empty_like(w)
        acc = 0.0
        for i in range(w.size):
            acc += a * (w[i] - acc)
            x[i] = acc
        sd = float(np.std(x))
        th = x * (A / sd) if sd > 0 else x
    else:  # pragma: no cover
        raise ValueError(sc)
    om = np.gradient(th, h)
    al = np.gradient(om, h)
    return th, om, al
