"""Модели сервопривода MG90S.

Общий конвейер (одинаков для всех моделей, каждое звено отключаемо):
    импульс в программе -> [кадр PWM: серва читает импульс раз за кадр] -> [задержка]
    -> целевой угол по калибровке -> динамика привода -> [люфт] -> [упоры] -> угол вала

Электромеханическая модель (по структуре реальной сервы: усилитель с насыщением -> DC-мотор -> редуктор ->
потенциометр на выходном валу, см. разборку SG90 в README):

    e   = theta_ref - theta_out                 (потенциометр измеряет ВЫХОДНОЙ вал)
    u   = 0 при |e| < deadband/2, иначе clip(e/e_sat, -1, 1)
    T_m = T_stall * (u - omega/omega_nl)        (линейная механическая характеристика DC-мотора)
    J * domega/dt = T_m - c_v*omega - T_coulomb*sign(omega) - T_grav - T_ext + T_inertial
    theta_out = backlash(theta_motor), затем механические упоры.

Из документации известны только T_stall (по питанию) и среднее время поворота на 60 градусов без нагрузки.
Остальное (tau_a, e_sat, люфт, трение, инерция) - параметры модели, а не данные MG90S; omega_nl по умолчанию
подгоняется так, чтобы модель повторяла паспортное время 60 градусов.
"""
from __future__ import annotations

import math
from collections import deque
from functools import lru_cache

DEG = math.pi / 180.0
KGCM_TO_NM = 0.0980665
SUPPLY_SPEC = {
    "4.8": {"t60_s": 0.10, "stall_kgcm": 1.8},
    "6.0": {"t60_s": 0.08, "stall_kgcm": 2.2},
}


class Calibration:
    """Соответствие импульс <-> угол вала (угол отсчитывается от центрального импульса)."""

    def __init__(self, p):
        self.p_center = float(p.center_pulse_us)
        self.slope = (p.cal_amax_deg - p.cal_amin_deg) / (p.cal_pmax_us - p.cal_pmin_us)  # град/мкс
        self.a_center = p.cal_amin_deg + (p.center_pulse_us - p.cal_pmin_us) * self.slope
        self.lo = p.mech_min_deg - self.a_center
        self.hi = p.mech_max_deg - self.a_center

    def pulse_to_angle(self, pulse: float) -> float:
        return (pulse - self.p_center) * self.slope

    def angle_to_pulse(self, angle: float) -> float:
        return self.p_center + angle / self.slope

    @property
    def us_per_deg(self) -> float:
        return 1.0 / self.slope


def stall_torque_nm(p) -> float:
    if p.stall_torque_nm > 0:
        return p.stall_torque_nm
    return SUPPLY_SPEC[p.supply]["stall_kgcm"] * KGCM_TO_NM


def _unloaded_t60(omega_nl_dps: float, tau_a_s: float, e_sat_deg: float, delta=60.0, tol=1.0, h=2e-4, t_max=1.5):
    """Время, за которое холостой вал электромеханической модели впервые достигает delta-tol (без доп. эффектов)."""
    th, w = 0.0, 0.0                      # град, град/с
    t = 0.0
    target = delta - tol
    while t < t_max:
        e = delta - th
        u = max(-1.0, min(1.0, e / e_sat_deg))
        w = (w + h * omega_nl_dps * u / tau_a_s) / (1.0 + h / tau_a_s)
        th += h * w
        t += h
        if th >= target:
            return t
    return float("inf")


@lru_cache(maxsize=256)
def calibrate_omega_nl(tau_a_ms: float, e_sat_deg: float, target_t60_s: float) -> float:
    """Подбор скорости холостого хода omega_nl [град/с] под паспортное время поворота на 60 градусов."""
    tau = tau_a_ms * 1e-3
    lo, hi = 100.0, 20000.0
    if _unloaded_t60(hi, tau, e_sat_deg) > target_t60_s:
        return hi            # при заданных tau_a и e_sat паспортное время недостижимо
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        if _unloaded_t60(mid, tau, e_sat_deg) > target_t60_s:
            lo = mid
        else:
            hi = mid
    return hi


def resolve_omega_nl(p) -> float:
    if p.auto_omega_nl:
        return calibrate_omega_nl(round(p.tau_a_ms, 6), round(p.e_sat_deg, 6), SUPPLY_SPEC[p.supply]["t60_s"])
    return float(p.omega_nl_dps)


def electromech_small_signal(p):
    """Малосигнальные характеристики внутреннего контура без нагрузки: omega_n [рад/с], zeta, K_pos [1/с]."""
    omega_nl = resolve_omega_nl(p)
    k_pos = omega_nl / p.e_sat_deg
    tau = p.tau_a_ms * 1e-3
    wn = math.sqrt(k_pos / tau)
    zeta = 1.0 / (2.0 * math.sqrt(k_pos * tau))
    return {"omega_nl_dps": omega_nl, "k_pos_per_s": k_pos, "omega_n_rad_s": wn, "zeta": zeta}


class ServoBase:
    name = "base"

    def __init__(self, p, h: float):
        self.p = p
        self.h = h
        self.cal = Calibration(p)
        self.enabled = bool(p.servo_enabled)
        self.frame_T = (1.0 / p.pwm_frame_hz) if p.en_frame else 0.0
        d = (p.delay_ms * 1e-3 / h) if p.en_delay else 0.0      # задержка в подшагах (дробная, линейная интерполяция)
        self.delay_int = int(math.floor(d))
        self.delay_frac = d - self.delay_int
        self.delay_steps = self.delay_int + (1 if self.delay_frac > 1e-9 else 0)
        self.lo, self.hi = (self.cal.lo, self.cal.hi) if p.en_stops else (-1e9, 1e9)
        self.db_deg = p.deadband_us * self.cal.slope if p.en_deadband else 0.0
        self.half_lash = 0.5 * p.backlash_deg if p.en_backlash else 0.0
        self.theta = 0.0            # угол выходного вала относительно основания, град
        self.theta_m = 0.0          # угол на стороне мотора (до люфта), град
        self.omega_dps = 0.0        # скорость выходного вала, град/с
        self.omega_rad = 0.0
        self.theta_ref = 0.0        # декодированный (после кадра и задержки) целевой угол, град
        self.at_stop = False
        self.held_pulse = 0.0
        self.next_frame = 0.0
        self.delay_buf: deque = deque()

    def reset(self, theta0_deg: float, pulse0_us: float):
        theta0_deg = min(max(float(theta0_deg), self.lo), self.hi)       # вал не может стартовать за упором
        self.theta = self.theta_m = theta0_deg
        self.omega_dps = self.omega_rad = 0.0
        self.held_pulse = float(pulse0_us)
        self.next_frame = 0.0
        ref0 = self.cal.pulse_to_angle(pulse0_us)
        self.theta_ref = ref0
        self.delay_buf = deque([ref0] * (self.delay_int + 2), maxlen=self.delay_int + 2)
        self.at_stop = False

    def _decode(self, t: float, pulse: float) -> float:
        if self.frame_T > 0.0:
            if t >= self.next_frame - 1e-12:
                self.held_pulse = pulse
                while self.next_frame <= t + 1e-12:
                    self.next_frame += self.frame_T
        else:
            self.held_pulse = pulse
        ref = self.cal.pulse_to_angle(self.held_pulse)
        if self.delay_steps > 0:
            buf = self.delay_buf
            buf.append(ref)
            d0 = self.delay_int
            ref = buf[-1 - d0] * (1.0 - self.delay_frac) + buf[-2 - d0] * self.delay_frac
        self.theta_ref = ref
        return ref

    def _finish(self, theta_m: float, w_rad: float):
        half = self.half_lash
        th = self.theta
        if half > 0.0:
            d = theta_m - th
            if d > half:
                th = theta_m - half
            elif d < -half:
                th = theta_m + half
        else:
            th = theta_m
        at_stop = False
        if th > self.hi:
            th = self.hi
            theta_m = min(theta_m, self.hi + half)
            if w_rad > 0.0:
                w_rad = 0.0
            at_stop = True
        elif th < self.lo:
            th = self.lo
            theta_m = max(theta_m, self.lo - half)
            if w_rad < 0.0:
                w_rad = 0.0
            at_stop = True
        self.omega_dps = (th - self.theta) / self.h
        self.theta = th
        self.theta_m = theta_m
        self.omega_rad = w_rad
        self.at_stop = at_stop

    def step(self, t: float, pulse: float, alpha_base_rad: float, theta_platform_deg: float):  # pragma: no cover
        raise NotImplementedError


class IdealServo(ServoBase):
    """Идеальный привод: положение мгновенно равно целевому (только для сравнения)."""
    name = "ideal"

    def step(self, t, pulse, alpha_base_rad, theta_platform_deg):
        ref = self.cal.pulse_to_angle(pulse)
        self.theta_ref = ref
        if not self.enabled:
            self.omega_dps = 0.0
            return
        th = min(max(ref, self.lo), self.hi)
        self.at_stop = th != ref
        self.omega_dps = (th - self.theta) / self.h
        self.theta = self.theta_m = th


class FirstOrderServo(ServoBase):
    """Упрощённый привод: апериодическое звено + ограничение скорости + кадры/задержка/мёртвая зона/люфт/упоры."""
    name = "first_order"

    def __init__(self, p, h):
        super().__init__(p, h)
        self.alpha = 1.0 - math.exp(-h / (p.fo_tau_ms * 1e-3))
        self.vlim = p.fo_vmax_dps * h

    def step(self, t, pulse, alpha_base_rad, theta_platform_deg):
        ref = self._decode(t, pulse)
        if not self.enabled:
            self.omega_dps = 0.0
            return
        e = ref - self.theta
        if abs(e) <= 0.5 * self.db_deg:
            d = 0.0
        else:
            d = e * self.alpha
            if d > self.vlim:
                d = self.vlim
            elif d < -self.vlim:
                d = -self.vlim
        self._finish(self.theta_m + d, d / self.h * DEG)


class ElectroMechServo(ServoBase):
    """Электромеханическая модель: насыщающий позиционный контур + DC-мотор + нагрузка (см. docstring модуля)."""
    name = "electromech"

    def __init__(self, p, h):
        super().__init__(p, h)
        self.stall = stall_torque_nm(p)
        self.omega_nl = resolve_omega_nl(p)                       # град/с
        self.tau_a = p.tau_a_ms * 1e-3
        self.e_sat = float(p.e_sat_deg)
        self.j_servo = self.tau_a * self.stall / (self.omega_nl * DEG)
        self.j_load = p.j_load_kgm2 * p.k_mech ** 2
        self.J = self.j_servo + self.j_load
        visc = p.viscous_nms if p.en_friction else 0.0
        self.c = self.stall / (self.omega_nl * DEG) + visc         # Н*м*с/рад
        self.tc = p.coulomb_nm if p.en_friction else 0.0
        self.grav = p.grav_nm
        self.ext = p.ext_torque_nm
        self.k = p.k_mech
        self.j_inertial = p.j_load_kgm2 * p.k_mech if p.en_inertial else 0.0

    def step(self, t, pulse, alpha_base_rad, theta_platform_deg):
        ref = self._decode(t, pulse)
        if not self.enabled:
            self.omega_dps = 0.0
            return
        h = self.h
        e = ref - self.theta
        if abs(e) <= 0.5 * self.db_deg:
            u = 0.0
        else:
            u = e / self.e_sat
            u = 1.0 if u > 1.0 else (-1.0 if u < -1.0 else u)
        drive = (self.stall * u - self.grav * math.sin(theta_platform_deg * DEG) * self.k
                 + self.ext - self.j_inertial * alpha_base_rad)
        w_free = (self.omega_rad + h * drive / self.J) / (1.0 + h * self.c / self.J)
        if self.tc > 0.0:
            d = h * self.tc / self.J
            w_new = 0.0 if abs(w_free) <= d else w_free - math.copysign(d, w_free)
        else:
            w_new = w_free
        self._finish(self.theta_m + h * w_new / DEG, w_new)


class LoadLash:
    """Люфт между выходным валом и платформой (потенциометр его не видит): платформа отстаёт при реверсе."""

    def __init__(self, p):
        self.half = 0.5 * p.backlash_load_deg if p.en_backlash else 0.0
        self.th = 0.0

    def reset(self, th0: float):
        self.th = float(th0)

    def apply(self, theta_out: float) -> float:
        if self.half <= 0.0:
            self.th = theta_out
            return theta_out
        d = theta_out - self.th
        if d > self.half:
            self.th = theta_out - self.half
        elif d < -self.half:
            self.th = theta_out + self.half
        return self.th


def make_servo(p, h: float) -> ServoBase:
    if p.servo_model == "ideal":
        return IdealServo(p, h)
    if p.servo_model == "first_order":
        return FirstOrderServo(p, h)
    return ElectroMechServo(p, h)


def max_substep(p) -> float:
    """Максимальный устойчивый подшаг интегрирования сервы."""
    h = 1e-3
    if p.servo_model == "electromech":
        sm = electromech_small_signal(p)
        h = min(h, 0.2 / sm["omega_n_rad_s"], 0.4 * p.tau_a_ms * 1e-3)
    elif p.servo_model == "first_order":
        h = min(h, 0.4 * p.fo_tau_ms * 1e-3)
    return max(h, 5e-5)


# ----------------------------------------------------------------------------- испытания сервы отдельно от контура
def servo_response(p, profile, t_end: float, h: float = 5e-4):
    """Отклик сервы на заданный профиль импульса profile(t) [мкс] при неподвижном основании."""
    servo = make_servo(p, h)
    pulse0 = profile(0.0)
    servo.reset(servo.cal.pulse_to_angle(pulse0), pulse0)
    n = int(round(t_end / h))
    t_arr, pulse_arr, cmd_arr, th_arr, w_arr, ref_arr, stop_arr = [], [], [], [], [], [], []
    for i in range(n + 1):
        t = i * h
        pulse = profile(t)
        t_arr.append(t)
        pulse_arr.append(pulse)
        cmd_arr.append(servo.cal.pulse_to_angle(pulse))
        th_arr.append(servo.theta)
        w_arr.append(servo.omega_dps)
        ref_arr.append(servo.theta_ref)
        stop_arr.append(servo.at_stop)
        servo.step(t, pulse, 0.0, servo.theta)
    return {"t": t_arr, "pulse": pulse_arr, "cmd_angle": cmd_arr, "angle": th_arr, "omega": w_arr,
            "ref_angle": ref_arr, "at_stop": stop_arr}


def time_to_reach(p, delta_deg: float = 60.0, tol: float = 1.0, t_end: float = 1.5, h: float = 5e-4,
                  t_step: float = 0.05) -> float:
    """Время от момента изменения команды в программе до достижения delta-tol градусов.

    Включает ожидание кадра PWM и задержку сервы, если эти эффекты включены в параметрах.
    """
    cal = Calibration(p)
    p1 = cal.angle_to_pulse(delta_deg)
    p0 = cal.p_center
    res = servo_response(p, lambda t: p1 if t >= t_step else p0, t_end, h)
    target = delta_deg - tol
    for t, th in zip(res["t"], res["angle"]):
        if t >= t_step and th >= target:
            return t - t_step
    return float("inf")
