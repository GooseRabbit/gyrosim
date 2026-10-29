"""Сравнение модели сервы с измерениями и подбор параметров.

Формат измерения - ступенька команды: время t (с), угол вала/платформы (град, относительный), импульс до и после,
момент подачи ступеньки. Угол измерения приводится к нулю по начальному участку.
При подборе кадры PWM отключены (их фаза в измерении неизвестна), их эффект поглощает параметр delay_ms.
Подбираются параметры электромеханической модели: tau_a, e_sat, omega_nl, delay_ms.
Проверено на СИНТЕТИЧЕСКИХ данных, сгенерированных той же структурой модели: это проверка алгоритма подбора,
а не доказательство, что модель верно описывает реальную MG90S.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import params as P
from .servo import servo_response


@dataclass
class StepDataset:
    t: np.ndarray            # с
    angle: np.ndarray        # град
    pulse_from: float        # мкс
    pulse_to: float          # мкс
    t_step: float = 0.0      # момент изменения команды на оси времени измерения, с


def _model_angle(flat: dict, ds: StepDataset, h: float = 1e-3):
    p, _ = P.validate(flat)
    prof = lambda t, a=ds.pulse_from, b=ds.pulse_to, ts=ds.t_step: b if t >= ts else a
    t_end = float(np.max(ds.t))
    r = servo_response(p, prof, t_end, h)
    tm = np.asarray(r["t"])
    am = np.asarray(r["angle"])
    am = am - am[0]
    return np.interp(ds.t, tm, am), tm, am


def prepare(ds: StepDataset, n_base: int = 5) -> StepDataset:
    base = float(np.mean(ds.angle[:max(1, n_base)]))
    return StepDataset(np.asarray(ds.t, float), np.asarray(ds.angle, float) - base, ds.pulse_from, ds.pulse_to, ds.t_step)


def compare(flat: dict, datasets: list[StepDataset]) -> list[dict]:
    out = []
    for ds in datasets:
        ds = prepare(ds)
        model, tm, am = _model_angle(flat, ds)
        err = model - ds.angle
        out.append({"t": ds.t, "measured": ds.angle, "model": model, "error": err,
                    "rmse": float(np.sqrt(np.mean(err ** 2))), "max_abs": float(np.max(np.abs(err)))})
    return out


FIT_KEYS = ("tau_a_ms", "e_sat_deg", "omega_nl_dps", "delay_ms")
FIT_BOUNDS = {"tau_a_ms": (1.0, 150.0), "e_sat_deg": (1.0, 100.0), "omega_nl_dps": (150.0, 4000.0), "delay_ms": (0.0, 80.0)}
_LOG_KEYS = ("tau_a_ms", "e_sat_deg", "omega_nl_dps")


def initial_guess(datasets: list[StepDataset]) -> dict:
    """Начальные значения по признакам записи: скорость - из наклона большого шага, задержка - из начала движения."""
    ds = max((prepare(d) for d in datasets), key=lambda d: float(np.max(np.abs(d.angle))))
    t, a = ds.t, ds.angle
    sign = 1.0 if a[np.argmax(np.abs(a))] >= 0 else -1.0
    a = sign * a
    k = max(3, len(t) // 60)
    v = np.gradient(a, t)
    v = np.convolve(v, np.ones(k) / k, mode="same")
    omega0 = float(np.clip(np.max(v), 200.0, 3000.0))
    amp = float(np.max(a))
    moved = np.where((t >= ds.t_step) & (a > max(0.3, 0.03 * amp)))[0]
    delay0 = float(np.clip((t[moved[0]] - ds.t_step) * 1e3, 0.0, 60.0)) if len(moved) else 0.0
    return {"omega_nl_dps": omega0, "delay_ms": delay0}


def fit_servo(flat: dict, datasets: list[StepDataset], free=FIT_KEYS, max_nfev: int = 120) -> dict:
    """Подбор параметров по МНК (несколько стартов, логарифмический масштаб). Возвращает параметры и ошибки."""
    from scipy.optimize import least_squares
    base = dict(flat)
    base.update({"servo_model": "electromech", "auto_omega_nl": False, "en_frame": False, "en_delay": True,
                 "en_deadband": False, "en_backlash": False, "en_friction": False, "j_load_kgm2": 0.0,
                 "en_inertial": False, "grav_nm": 0.0, "ext_torque_nm": 0.0})
    prepped = [prepare(d) for d in datasets]
    free = list(free)
    guess = initial_guess(datasets)
    lo = np.array([FIT_BOUNDS[k][0] for k in free])
    hi = np.array([FIT_BOUNDS[k][1] for k in free])
    islog = np.array([k in _LOG_KEYS for k in free])

    def to_x(vals):
        v = np.clip(np.asarray(vals, float), lo * 1.001 + 1e-9, hi * 0.999)
        return np.where(islog, np.log(np.maximum(v, 1e-9)), v)

    def from_x(x):
        return np.where(islog, np.exp(x), x)

    xlo, xhi = lo.copy(), hi.copy()
    xlo[islog] = np.log(lo[islog])
    xhi[islog] = np.log(hi[islog])

    def residuals(x):
        f = dict(base)
        for k, v in zip(free, from_x(x)):
            f[k] = float(v)
        r = []
        for ds in prepped:
            model, _, _ = _model_angle(f, ds)
            r.append(model - ds.angle)
        return np.concatenate(r)

    starts = []
    for tau_mul in (1.0, 2.5, 0.4):
        for esat in (15.0, 40.0):
            start = {"tau_a_ms": float(base["tau_a_ms"]) * tau_mul, "e_sat_deg": esat,
                     "omega_nl_dps": guess["omega_nl_dps"], "delay_ms": guess["delay_ms"]}
            starts.append([start[k] if k in start else float(base[k]) for k in free])
    best = None
    first_rmse = None
    for s0 in starts:
        x0 = to_x(s0)
        try:
            sol = least_squares(residuals, x0, bounds=(xlo, xhi), diff_step=1e-2, max_nfev=max_nfev, x_scale=1.0)
        except Exception:  # noqa: BLE001 - пропускаем неудачный старт
            continue
        rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
        if first_rmse is None:
            first_rmse = float(np.sqrt(np.mean(residuals(x0) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, sol)
    if best is None:
        raise RuntimeError("Подбор не удался: проверьте данные")
    rmse, sol = best
    fitted = dict(base)
    vals = from_x(sol.x)
    for k, v in zip(free, vals):
        fitted[k] = float(v)
    return {"params": {k: fitted[k] for k in free}, "flat": fitted, "rmse_before": first_rmse, "rmse_after": rmse,
            "nfev": int(sol.nfev), "success": bool(sol.success)}


def model_step_characteristics(flat: dict, pulse_from: float, pulse_to: float, t_step: float = 0.05,
                               t_end: float = 1.2) -> dict:
    """Характеристики ступенчатого отклика модели для сравнения со сводкой измерений пользователя."""
    p, _ = P.validate(flat)
    r = servo_response(p, lambda t: pulse_to if t >= t_step else pulse_from, t_end, 5e-4)
    t = np.asarray(r["t"])
    a = np.asarray(r["angle"])
    a = a - a[0]
    n = len(a)
    final = float(np.mean(a[-max(5, n // 20):]))
    amp = abs(final)
    if amp < 1e-6:
        return {"amplitude_deg": 0.0}
    sgn = 1.0 if final > 0 else -1.0
    x = sgn * a
    after = t >= t_step

    def first(cond):
        idx = np.where(after & cond)[0]
        return float(t[idx[0]] - t_step) if len(idx) else None

    out = {"amplitude_deg": final,
           "start_delay_s": first(x > max(0.2, 0.02 * amp)),
           "reach_95_s": first(x >= 0.95 * amp),
           "t_10_90_s": None,
           "max_speed_dps": float(np.max(np.abs(np.gradient(a, t)))),
           "overshoot_pct": max(0.0, (float(np.max(x)) - amp) / amp * 100.0)}
    t10, t90 = first(x >= 0.1 * amp), first(x >= 0.9 * amp)
    if t10 is not None and t90 is not None:
        out["t_10_90_s"] = t90 - t10
    outside = np.where(np.abs(x - amp) > 0.02 * amp)[0]
    out["settle_2pct_s"] = None if len(outside) == 0 else float(t[min(outside[-1] + 1, n - 1)] - t_step)
    return out


def load_measurement_csv(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Два числовых столбца: время (с), угол (град). Разделители: ';', ',', табуляция, пробел; десятичная запятая допустима
    (если в строке есть ';' или табуляция, запятая считается десятичной). Заголовок и пустые строки пропускаются."""
    t, a = [], []
    for line in text.splitlines():
        if ";" in line:
            parts = [s.strip().replace(",", ".") for s in line.split(";")]
        elif "\t" in line:
            parts = [s.strip().replace(",", ".") for s in line.split("\t")]
        elif "," in line:
            parts = [s.strip() for s in line.split(",")]
        else:
            parts = [s.strip().replace(",", ".") for s in line.split()]
        parts = [x for x in parts if x != ""]
        if len(parts) < 2:
            continue
        try:
            t.append(float(parts[0]))
            a.append(float(parts[1]))
        except ValueError:
            continue
    if len(t) < 5:
        raise ValueError("В файле меньше 5 числовых строк (нужны столбцы: время в секундах, угол в градусах)")
    t = np.asarray(t)
    a = np.asarray(a)
    if np.any(np.diff(t) <= 0):
        raise ValueError("Время должно строго возрастать")
    return t, a
