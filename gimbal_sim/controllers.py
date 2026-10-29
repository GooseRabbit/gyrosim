"""Регуляторы и формирование команды как в скетче.

Знак и единицы:  correction_us - коррекция импульса от центра (мкс).
    P  : correction = -KP * angle
    PD : correction = -(KP * angle + KD * rate)
    Каскад: omega_ref = clip(-Ktheta * angle, +-omega_ref_max)
        скоростная форма:   d(correction)/dt = Kw * (omega_ref - rate)      (интегрирует ошибку скорости)
        пропорциональная:   correction = Kw * (omega_ref - rate)            (= PD с KP = Kw*Ktheta, KD = Kw)
    target_pulse = center + s_cmd * correction, затем ограничение [pmin, pmax] и |коррекция| <= MAX_CORRECTION.
"""
from __future__ import annotations

import math


def lroundf(x: float) -> int:
    """Округление как lroundf в C: половина от нуля."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


class Controller:
    def __init__(self, p):
        self.p = p
        self.s_cmd = -1.0 if p.invert_cmd else 1.0
        c = p.center_pulse_us
        lo, hi = p.pmin_sw_us, p.pmax_sw_us
        if p.enable_cmd_saturation:
            lo = max(lo, c - p.max_correction_us)
            hi = min(hi, c + p.max_correction_us)
        self.pulse_lo, self.pulse_hi = lo, hi
        # допустимая область состояния интегратора (если anti-windup включён)
        if p.enable_cmd_saturation:
            a_lo, a_hi = lo, hi
        else:
            a_lo, a_hi = p.cal_pmin_us, p.cal_pmax_us
        if self.s_cmd > 0:
            self.corr_lo, self.corr_hi = a_lo - c, a_hi - c
        else:
            self.corr_lo, self.corr_hi = -(a_hi - c), -(a_lo - c)
        self.corr_state = 0.0
        self.omega_ref = 0.0
        self.e_theta = 0.0
        self.e_omega = 0.0
        self.saturated = False
        self.updates = 0

    def reset(self, pulse0_us: float):
        # безударный старт: состояние интегратора соответствует текущему положению импульса
        self.corr_state = self.s_cmd * (pulse0_us - self.p.center_pulse_us)
        self.omega_ref = self.e_theta = self.e_omega = 0.0
        self.saturated = False
        self.updates = 0

    def update(self, angle: float, rate: float, dt: float):
        p = self.p
        self.e_theta = -angle
        self.e_omega = 0.0
        self.omega_ref = 0.0
        mode = p.ctrl_mode
        if mode == "P":
            corr = -p.kp_us_per_deg * angle
        elif mode == "PD":
            corr = -(p.kp_us_per_deg * angle + p.kd_us_per_dps * rate)
        else:
            w_ref = -p.ktheta_per_s * angle
            lim = p.omega_ref_max_dps
            w_ref = lim if w_ref > lim else (-lim if w_ref < -lim else w_ref)
            self.omega_ref = w_ref
            self.e_omega = w_ref - rate
            if p.cascade_inner == "velocity":
                self.corr_state += p.kw_us_per_dps * self.e_omega * dt
                if p.antiwindup:
                    self.corr_state = min(max(self.corr_state, self.corr_lo), self.corr_hi)
                corr = self.corr_state
            else:
                corr = p.kw_us_per_dps * self.e_omega
        pulse = p.center_pulse_us + self.s_cmd * corr
        sat = False
        if p.enable_cmd_saturation:
            if pulse > self.pulse_hi:
                pulse, sat = self.pulse_hi, True
            elif pulse < self.pulse_lo:
                pulse, sat = self.pulse_lo, True
        self.saturated = sat
        self.updates += 1
        return lroundf(pulse), sat


class PulseLimiter:
    """Повторяет updateServoMotion() из скетча: за период не более step мкс, затем constrain(min, max)."""

    def __init__(self, p):
        self.enabled = p.rate_limit_enable
        self.step_us = int(round(p.rate_step_us))
        self.period = p.rate_period_ms * 1e-3
        self.lo = int(round(p.pmin_sw_us))
        self.hi = int(round(p.pmax_sw_us))
        self.current = 0
        self.target = 0
        self.next_t = 0.0
        self.active = False

    def reset(self, pulse0: int):
        self.current = self.target = int(pulse0)
        self.next_t = self.period          # lastServoMoveTime = 0 при старте
        self.active = False

    def set_target(self, target: int):
        self.target = int(target)

    def step(self, t: float):
        if not self.enabled:
            self.current = self.target
            self.active = False
            return self.current, False
        if t >= self.next_t - 1e-12:
            self.next_t = t + self.period
            diff = self.target - self.current
            if diff > self.step_us:
                diff = self.step_us
            elif diff < -self.step_us:
                diff = -self.step_us
            if diff != 0:
                self.current = min(max(self.current + diff, self.lo), self.hi)
        self.active = abs(self.target - self.current) > self.step_us
        return self.current, self.active
