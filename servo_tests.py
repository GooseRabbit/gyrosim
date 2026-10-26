"""Испытания сервопривода отдельно от контура (раздел «поведение сервопривода» требований)."""
from __future__ import annotations

import copy
import math

import numpy as np

from .controllers import PulseLimiter
from .servo import Calibration, make_servo, resolve_omega_nl, servo_response, stall_torque_nm, time_to_reach


def _profile(points):
    """Кусочно-постоянный профиль: points = [(t, pulse), ...] по возрастанию t."""
    def f(t):
        v = points[0][1]
        for tt, pp in points:
            if t >= tt:
                v = pp
        return v
    return f


def _limit_chain(p, profile, t_end, h=5e-4):
    """Целевой импульс -> программный ограничитель -> серва: возвращает ряды как в испытании сервы."""
    servo = make_servo(p, h)
    lim = PulseLimiter(p)
    pulse0 = int(round(profile(0.0)))
    servo.reset(servo.cal.pulse_to_angle(pulse0), pulse0)
    lim.reset(pulse0)
    n = int(round(t_end / h))
    rows = {k: [] for k in ("t", "target", "pulse", "cmd_angle", "angle", "omega", "at_stop")}
    for i in range(n + 1):
        t = i * h
        lim.set_target(int(round(profile(t))))
        cur, _ = lim.step(t)
        for k, v in (("t", t), ("target", lim.target), ("pulse", cur), ("cmd_angle", servo.cal.pulse_to_angle(cur)),
                     ("angle", servo.theta), ("omega", servo.omega_dps), ("at_stop", servo.at_stop)):
            rows[k].append(v)
        servo.step(t, cur, 0.0, servo.theta)
    return rows


def _speed_bound(p) -> float:
    """Верхняя граница скорости вала для проверки «нет телепортации», град/с."""
    if p.servo_model == "first_order":
        return p.fo_vmax_dps
    if p.servo_model == "electromech":
        return resolve_omega_nl(p) * 1.001
    return float("inf")


def _checks(p, res, cal, h):
    th = np.asarray(res["angle"])
    checks = []
    lo, hi = (cal.lo, cal.hi) if p.en_stops else (-1e9, 1e9)
    checks.append(("угол не выходит за механические упоры", bool(np.all(th <= hi + 1e-6) and np.all(th >= lo - 1e-6))))
    vmax = _speed_bound(p)
    if math.isfinite(vmax):
        dv = np.max(np.abs(np.diff(th))) / h if len(th) > 1 else 0.0
        checks.append((f"скорость вала не превышает {vmax:.0f} град/с (фактически {dv:.0f})", bool(dv <= vmax * 1.02 + 1e-6)))
    else:
        checks.append(("скорость вала не проверяется для идеальной модели", True))
    checks.append(("нет нечисловых значений", bool(np.all(np.isfinite(th)))))
    return checks


def run_suite(p) -> list[dict]:
    """Возвращает список испытаний: name, desc, res (ряды) и checks [(текст, ok)]."""
    cal = Calibration(p)
    us = cal.angle_to_pulse
    c = cal.p_center
    h = 5e-4
    tests = []

    def add(name, desc, res, extra=None):
        tests.append({"name": name, "desc": desc, "res": res, "checks": _checks(p, res, cal, h) + (extra or [])})

    r = servo_response(p, _profile([(0, c), (0.05, us(60))]), 0.5, h)
    add("Большой шаг +60°", "Ступенька команды из центра на +60° (без программного ограничителя скорости).", r)
    r = servo_response(p, _profile([(0, c), (0.05, us(3))]), 0.5, h)
    add("Малый шаг +3°", "Малое изменение команды - малое перемещение; видна мёртвая зона и кадр PWM.", r)
    r = servo_response(p, _profile([(0, us(-60)), (0.05, us(60))]), 0.6, h)
    add("Большой переход -60° -> +60°", "Движение через весь рабочий диапазон: разгон, участок насыщения, торможение.", r)
    r = servo_response(p, _profile([(0, c), (0.05, us(20)), (0.35, us(-20)), (0.65, c)]), 1.0, h)
    add("Реверс +20° -> -20° -> 0", "Смена направления; при включённом люфте ширина люфта видна как задержка при реверсе.", r)

    # предел хода: сужаем упоры, команда за упор
    ps = copy.copy(p)
    ps.en_stops = True
    ps.mech_max_deg = cal.a_center + 45.0
    ps.mech_min_deg = cal.a_center - 45.0
    cal_s = Calibration(ps)
    r = servo_response(ps, _profile([(0, c), (0.05, us(80))]), 0.6, h)
    ok_stop = bool(max(r["angle"]) <= cal_s.hi + 1e-6)
    tests.append({"name": "Предел хода (упоры ±45°, команда +80°)", "desc": "Команда за упором не приводит к физически невозможному углу.",
                  "res": r, "checks": _checks(ps, r, cal_s, h) + [("вал остановлен упором, достигнут +45°", ok_stop and max(r["angle"]) > cal_s.hi - 1.0)]})

    # программный ограничитель скорости команды: вкл/выкл
    for state in (True, False):
        pl = copy.copy(p)
        pl.rate_limit_enable = state
        rows = _limit_chain(pl, _profile([(0, c), (0.1, c + 300.0)]), 1.2, h)
        res = {"t": rows["t"], "pulse": rows["pulse"], "cmd_angle": rows["cmd_angle"], "angle": rows["angle"],
               "omega": rows["omega"], "ref_angle": rows["cmd_angle"], "at_stop": rows["at_stop"], "target": rows["target"]}
        extra = []
        if state:
            d = np.abs(np.diff(np.asarray(rows["pulse"])))
            step_ok = bool(np.max(d) <= pl.rate_step_us + 1e-9)
            extra.append((f"скачок команды не больше {pl.rate_step_us:g} мкс за {pl.rate_period_ms:g} мс", step_ok))
        add("Ограничитель скорости PWM " + ("ВКЛ" if state else "ВЫКЛ"),
            "Шаг команды +300 мкс. С ограничителем команда нарастает со скоростью "
            f"{p.rate_step_us / (p.rate_period_ms * 1e-3):.0f} мкс/с (программная мера, не скорость вала).", res, extra=extra)

    if p.servo_model == "electromech":
        pl = copy.copy(p)
        pl.j_load_kgm2 = max(p.j_load_kgm2, 3e-4) * 3.0
        pl.en_friction = True
        pl.coulomb_nm = max(p.coulomb_nm, 0.01)
        heavy = servo_response(pl, _profile([(0, c), (0.05, us(60))]), 0.8, h)
        t_l = time_to_reach(p, 60.0, 1.0)
        t_h = time_to_reach(pl, 60.0, 1.0)
        add(f"Нагрузка: без нагрузки и с J={pl.j_load_kgm2:.1e} кг*м² + сухое трение",
            f"Время достижения 59° растёт с нагрузкой: {t_l:.3f} с -> {t_h:.3f} с.", heavy,
            extra=[("с нагрузкой серва не быстрее, чем без неё", bool(t_h >= t_l - 1e-6))])
        # недостаточный момент: противодействующий момент 0.6 от стартового. Мотор уравновешивает его при u = 0.6,
        # то есть при ошибке 0.6*e_sat: серва останавливается, не дойдя до цели.
        pw = copy.copy(p)
        pw.ext_torque_nm = -0.6 * stall_torque_nm(p)
        r_w = servo_response(pw, _profile([(0, c), (0.05, us(60))]), 1.0, h)
        final = r_w["angle"][-1]
        expected_short = 0.6 * p.e_sat_deg
        add("Недостаточный момент (противодействие 0.6 от стартового)",
            f"Серва останавливается в {final:.1f}° вместо 60°: ожидаемая недоходка около 0.6*e_sat = {expected_short:.1f}°.", r_w,
            extra=[("цель не достигнута при недостаточном моменте", final < 59.0),
                   ("недоходка близка к 0.6*e_sat (допуск 2.5°)", abs((60.0 - final) - expected_short) < 2.5)])

    # люфт внутри контура: реверс задерживается, но выходной вал доходит до цели (потенциометр на выходном валу)
    pb = copy.copy(p)
    pb.en_backlash = True
    pb.backlash_deg = max(p.backlash_deg, 3.0)
    prof = _profile([(0, us(10)), (0.05, us(10 - 2.0))])
    r_l = servo_response(pb, prof, 0.6, h)
    pn = copy.copy(p)
    pn.en_backlash = False
    r_n = servo_response(pn, prof, 0.6, h)

    def t_reach(res, target, tol=0.3):
        for tt, th in zip(res["t"], res["angle"]):
            if tt > 0.05 and th <= target + tol:
                return tt - 0.05
        return float("inf")
    tl, tn = t_reach(r_l, 8.0), t_reach(r_n, 8.0)
    add("Люфт 3° внутри контура: реверс на 2°",
        f"Реверс малой ступеньки: время до цели {tn * 1e3:.0f} мс без люфта и {tl * 1e3:.0f} мс с люфтом (мотор сначала проходит люфт).",
        r_l, extra=[("люфт задерживает реверс, но вал доходит до цели", tl > tn and tl < 0.5)])
    return tests
