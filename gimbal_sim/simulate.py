"""Замкнутый контур: основание + серва + платформа/IMU + модель измерения + регулятор.

Кинематика (одна ось, град):
    theta_platform = theta_base + k_mech * theta_load (+ постоянное смещение только если k_mech ~ 0)
    theta_IMU (истинный) = theta_platform        -> IMU закреплена на платформе, которую двигает серва
    theta_load = угол выходного вала после люфта нагрузки (если люфт нагрузки выключен, равен углу вала)
Порядок расчёта на каждом подшаге:
    1. IMU видит текущую ориентацию (с задержкой/шумом/частотой пакетов)  -> последний пакет
    2. регулятор (по своему периоду) считает целевой импульс по ПОСЛЕДНЕМУ пакету
    3. программный ограничитель скорости команды (если включён)
    4. модель сервы обрабатывает фактический импульс -> фактический угол вала
    5. из фактического угла вала и угла основания получается новая ориентация платформы/IMU
Целевой угол сервы никогда не подставляется вместо фактического.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from . import params as P
from .controllers import Controller, PulseLimiter
from .imu import IMUModel
from .scenarios import base_signal
from .servo import DEG, Calibration, LoadLash, electromech_small_signal, make_servo, max_substep


class SimulationError(RuntimeError):
    pass


@dataclass
class SimResult:
    t: np.ndarray
    s: dict
    params: SimpleNamespace
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


SERIES_KEYS = [
    "theta_true", "theta_meas", "omega_true", "omega_meas", "omega_ref", "e_theta", "e_omega",
    "target_pulse", "actual_pulse", "cmd_angle", "servo_ref_angle", "servo_angle", "servo_omega",
    "base_angle", "base_rate", "cmd_sat", "rate_active", "at_stop", "load_angle",
]


def loop_gain_info(p) -> dict:
    """Статическая (квазистатическая) оценка знака и величины усиления петли для P-режима."""
    cal = Calibration(p)
    s_m = -1.0 if p.invert_meas else 1.0
    s_c = -1.0 if p.invert_cmd else 1.0
    k_eff = p.kp_us_per_deg * cal.slope * p.k_mech * s_c * s_m   # >0 - отрицательная обратная связь
    info = {
        "us_per_deg": cal.us_per_deg,
        "K_eff_P": k_eff,
        "feedback": "отрицательная (компенсирует)" if k_eff > 0 else ("положительная (усиливает отклонение)" if k_eff < 0 else "нет"),
        "residual_fraction_P": (1.0 / (1.0 + k_eff)) if k_eff > -1.0 else float("inf"),
    }
    return info


def simulate(flat: dict) -> SimResult:
    p, warns = P.validate(flat)
    dt = p.dt_ms * 1e-3
    h_max = max_substep(p)
    n_sub = max(1, min(50, int(math.ceil(dt / h_max - 1e-9))))
    h = dt / n_sub
    n_out = int(round(p.duration_s / dt))
    N = n_out * n_sub
    if N > 3_000_000:
        raise SimulationError("Слишком много шагов: уменьшите длительность или увеличьте шаг расчёта")

    cal = Calibration(p)
    k = p.k_mech
    notes: list[str] = []

    # --- начальное состояние: платформа отклонена, это НЕ постоянное возмущение
    e0 = p.initial_error_deg
    offset = 0.0
    if abs(k) >= 0.01:
        th_out0 = e0 / k
        if p.en_stops and (th_out0 > cal.hi or th_out0 < cal.lo):
            th_out0 = min(max(th_out0, cal.lo), cal.hi)
            notes.append(f"Начальная ошибка {e0:g} град недостижима для вала; принято {k * th_out0:g} град.")
    else:
        th_out0 = 0.0
        offset = e0
        notes.append("k_mech около нуля: начальная ошибка сохраняется как постоянное смещение платформы.")

    TH, OM, AL = base_signal(p, N, h)
    servo = make_servo(p, h)
    imu = IMUModel(p, h)
    ctrl = Controller(p)
    lim = PulseLimiter(p)
    lash = LoadLash(p)

    pulse0 = int(round(cal.angle_to_pulse(th_out0)))
    servo.reset(th_out0, pulse0)
    th_out0 = servo.theta                                   # после обрезки упорами
    lash.reset(th_out0)
    lash_prev = th_out0
    theta_p0 = TH[0] + k * th_out0 + offset
    imu.reset(theta_p0, 0.0)
    ctrl.reset(pulse0)
    lim.reset(pulse0)

    T_c = p.control_period_s
    next_ctrl = 0.0 if p.first_update_immediate else T_c
    target_pulse = pulse0
    cmd_sat = False

    out = {key: np.zeros(n_out + 1) for key in SERIES_KEYS}
    oi = 0
    rate_any = 0.0
    for i in range(N + 1):
        t = i * h
        thb = TH[i]
        th_eff = lash.apply(servo.theta)
        w_eff = servo.omega_dps if lash.half <= 0.0 else (th_eff - lash_prev) / h
        lash_prev = th_eff
        theta_p = thb + k * th_eff + offset
        omega_p = OM[i] + k * w_eff
        ang_m, rate_m, _ = imu.update(t, theta_p, omega_p)
        if t >= next_ctrl - 1e-12:
            target_pulse, cmd_sat = ctrl.update(ang_m, rate_m, T_c)
            lim.set_target(target_pulse)
            next_ctrl += T_c
        cur, active = lim.step(t)
        if active:
            rate_any = 1.0
        if i % n_sub == 0:
            out["theta_true"][oi] = theta_p
            out["theta_meas"][oi] = ang_m
            out["omega_true"][oi] = omega_p
            out["omega_meas"][oi] = rate_m
            out["omega_ref"][oi] = ctrl.omega_ref
            out["e_theta"][oi] = -ang_m                      # ошибка по последнему пакету (цель 0)
            out["e_omega"][oi] = (ctrl.omega_ref - rate_m) if p.ctrl_mode == "cascade" else 0.0
            out["target_pulse"][oi] = target_pulse
            out["actual_pulse"][oi] = cur
            out["cmd_angle"][oi] = cal.pulse_to_angle(cur)
            out["servo_ref_angle"][oi] = servo.theta_ref
            out["servo_angle"][oi] = servo.theta
            out["servo_omega"][oi] = servo.omega_dps
            out["base_angle"][oi] = thb
            out["base_rate"][oi] = OM[i]
            out["cmd_sat"][oi] = 1.0 if cmd_sat else 0.0
            out["rate_active"][oi] = rate_any
            out["at_stop"][oi] = 1.0 if servo.at_stop else 0.0
            out["load_angle"][oi] = th_eff
            rate_any = 0.0
            oi += 1
        if i == N:
            break
        servo.step(t, cur, AL[i] * DEG, theta_p)

    for key, arr in out.items():
        if not np.all(np.isfinite(arr)):
            raise SimulationError(f"В ряду «{key}» появились нечисловые значения: проверьте параметры")

    t_arr = np.arange(n_out + 1) * dt
    info = loop_gain_info(p)
    info.update({
        "dt_s": dt, "substep_s": h, "n_sub": n_sub, "packets": imu.packets, "controller_updates": ctrl.updates,
        "control_period_s": T_c, "servo_model": p.servo_model,
    })
    if p.servo_model == "electromech":
        sm = electromech_small_signal(p)
        info.update({"omega_nl_dps": sm["omega_nl_dps"], "zeta_internal": sm["zeta"],
                     "omega_n_internal": sm["omega_n_rad_s"]})
    all_warn = list(warns) + notes
    return SimResult(t=t_arr, s=out, params=p, warnings=all_warn, info=info)


def simulate_kp_list(flat: dict, kp_list) -> list[SimResult]:
    """Одно и то же испытание для нескольких KP (остальные параметры одинаковы)."""
    res = []
    for kp in kp_list:
        f = dict(flat)
        f["kp_us_per_deg"] = float(kp)
        res.append(simulate(f))
    return res
