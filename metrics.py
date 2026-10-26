"""Показатели качества. Ошибка = ИСТИННЫЙ угол платформы (цель всегда 0 градусов)."""
from __future__ import annotations

import math

import numpy as np


def zigzag_extremes(x: np.ndarray, thr: float):
    """Чередующиеся экстремумы с порогом разворота thr (устойчив к плоским вершинам и шуму). Индексы."""
    n = len(x)
    if n < 3:
        return []
    ext = []
    hi_i = lo_i = 0
    direction = 0
    for i in range(1, n):
        if direction >= 0 and x[i] > x[hi_i]:
            hi_i = i
        if direction <= 0 and x[i] < x[lo_i]:
            lo_i = i
        if direction >= 0 and x[hi_i] - x[i] > thr and hi_i >= lo_i:
            if direction == 0 and lo_i < hi_i and x[hi_i] - x[lo_i] > thr:
                ext.append(lo_i)
            ext.append(hi_i)
            direction = -1
            lo_i = i
        elif direction <= 0 and x[i] - x[lo_i] > thr and lo_i >= hi_i:
            if direction == 0 and hi_i < lo_i and x[hi_i] - x[lo_i] > thr:
                ext.append(hi_i)
            ext.append(lo_i)
            direction = 1
            hi_i = i
    return ext


def classify_oscillation(t, x, sat_flags=None, band=1.0) -> dict:
    """Различает: нет колебаний / затухающие / устойчивые (предельный цикл) / нарастающие, по половинным размахам."""
    n = len(x)
    out = {"label": "нет данных", "detail": "", "period_s": None, "growth": None, "amplitude": None, "reliable": False}
    if n < 10:
        return out
    i0 = int(0.4 * n)
    xs = np.asarray(x[i0:])
    ts = np.asarray(t[i0:])
    peak = float(np.max(np.abs(x)))
    thr = max(0.2, 0.02 * peak)
    ext = zigzag_extremes(xs, thr)
    if len(ext) < 4:
        if abs(x[-1]) > band:
            out["label"] = "вывод невозможен: ошибка ещё не вышла в полосу"
            out["detail"] = "На рассматриваемом участке нет полных колебаний, а ошибка превышает полосу установления. " \
                            "Увеличьте длительность симуляции."
        else:
            out["label"] = "колебаний нет"
            out["detail"] = "На втором участке записи не найдено разворотов сигнала крупнее порога " \
                            f"{thr:.2g} град; ошибка в полосе установления."
            out["reliable"] = True
        return out
    vals = xs[ext]
    amps = np.abs(np.diff(vals)) / 2.0
    times = ts[ext]
    half_periods = np.diff(times)
    period = 2.0 * float(np.median(half_periods)) if len(half_periods) else None
    j = np.arange(len(amps))
    amps_pos = np.maximum(amps, 1e-9)
    slope = float(np.polyfit(j, np.log(amps_pos), 1)[0])
    g = math.exp(slope)                         # изменение размаха за половину периода
    out.update({"period_s": period, "growth": g, "amplitude": float(np.median(amps)),
                "reliable": len(amps) >= 6})
    amp_med = float(np.median(amps))
    if amp_med < max(0.3, 0.01 * peak):
        out["label"] = "малые колебания на уровне порога"
        out["detail"] = f"Размах около {amp_med:.2g} град (порог шума {thr:.2g}); устойчивых колебаний нет."
        return out
    sat_part = 0.0
    if sat_flags is not None:
        sat_part = float(np.mean(np.asarray(sat_flags)[i0:]))
    if g > 1.08:
        label = "нарастающие колебания"
    elif g < 0.92:
        label = "затухающие колебания"
    else:
        label = "устойчивые колебания (предельный цикл)"
    if sat_part > 0.2 and g <= 1.08:
        label += ", ограничены насыщением"
    out["label"] = label
    out["detail"] = (f"Размах {amp_med:.3g} град, период ~{period:.3g} с, изменение размаха за половину периода "
                     f"x{g:.3f}; найдено {len(amps)} половинных колебаний."
                     + ("" if out["reliable"] else " Мало колебаний на участке: вывод ненадёжен, увеличьте длительность."))
    return out


def _zero_crossings(x, hyst):
    state = 0
    count = 0
    for v in x:
        if state >= 0 and v < -hyst:
            if state == 1:
                count += 1
            state = -1
        elif state <= 0 and v > hyst:
            if state == -1:
                count += 1
            state = 1
        elif state == 0 and abs(v) > hyst:
            state = 1 if v > 0 else -1
    return count


def compute_metrics(res) -> list[dict]:
    """Список строк: key, label, value (число/текст/None), unit, help."""
    p = res.params
    t = res.t
    e = res.s["theta_true"]
    e0 = float(e[0])
    n = len(e)
    band = p.band_deg
    rows = []

    def add(key, label, value, unit, help_text):
        rows.append({"key": key, "label": label, "value": value, "unit": unit, "help": help_text})

    max_abs = float(np.max(np.abs(e)))
    add("max_abs", "Максимальный модуль ошибки", max_abs, "град", "Наибольшее отклонение платформы от нуля за всё испытание.")
    add("final_err", "Остаточная ошибка в конце", float(e[-1]), "град",
        "Угол платформы в последний момент. При колебаниях это значение случайно: смотрите классификацию колебаний.")
    add("rmse", "Среднеквадратичная ошибка (RMSE)", float(np.sqrt(np.mean(e ** 2))), "град",
        "Корень из среднего квадрата ошибки за всё испытание; учитывает и начальную ошибку.")
    outside = np.where(np.abs(e) > band)[0]
    if len(outside) == 0:
        ts = 0.0
    elif outside[-1] >= n - 1:
        ts = None
    else:
        ts = float(t[outside[-1] + 1])
    add("settle", f"Время установления в полосу ±{band:g}°", ts, "с",
        "Момент, после которого ошибка больше не выходит из полосы. «—» означает, что в конце испытания ошибка вне полосы.")
    if abs(e0) >= 0.5:
        sgn = 1.0 if e0 > 0 else -1.0
        over = max(0.0, -float(np.min(sgn * e))) / abs(e0) * 100.0
        add("overshoot", "Максимальное перерегулирование", over, "%",
            "Насколько платформа проскочила ноль в противоположную сторону, в процентах от начальной ошибки.")
    else:
        add("overshoot", "Максимальное перерегулирование", None, "%",
            "Не определено: начальная ошибка близка к нулю (используйте максимальный модуль ошибки).")
    zc = _zero_crossings(e, max(0.05, 0.002 * max_abs))
    add("zero_cross", "Число пересечений нуля", zc, "",
        "Сколько раз угол платформы менял знак (с гистерезисом против шума). Много пересечений - раскачка.")
    osc = classify_oscillation(t, e, res.s["cmd_sat"] + res.s["at_stop"], band)
    add("osc", "Колебания", osc["label"], "", osc["detail"] or "Классификация по размахам последних 60 % записи.")
    pulses = res.s["actual_pulse"]
    add("pulse_max", "Наибольший достигнутый импульс", float(np.max(pulses)), "мкс", "Фактический импульс после ограничителя.")
    add("pulse_min", "Наименьший достигнутый импульс", float(np.min(pulses)), "мкс", "Фактический импульс после ограничителя.")
    add("sat_cmd", "Доля времени в насыщении команды", float(np.mean(res.s["cmd_sat"])) * 100.0, "%",
        "Команда упиралась в пределы импульса или MAX_CORRECTION: линейный закон в это время не работает.")
    add("sat_stop", "Доля времени на механическом упоре", float(np.mean(res.s["at_stop"])) * 100.0, "%",
        "Вал находился на упоре: регулятор не может двигать платформу дальше.")
    add("rate_lim", "Доля времени с ограничением скорости команды", float(np.mean(res.s["rate_active"])) * 100.0, "%",
        "Программный ограничитель (10 мкс за 20 мс) отрезал часть требуемого изменения команды.")
    add("servo_vmax", "Максимальная угловая скорость вала", float(np.max(np.abs(res.s["servo_omega"]))), "град/с",
        "Наибольшая скорость выходного вала сервы за испытание.")
    # знак обратной связи
    fb = res.info.get("feedback", "")
    add("feedback", "Знак обратной связи (P, статически)", fb, "",
        f"Статическое усиление петли K = KP * (град вала на мкс) * k_mech со знаками = {res.info.get('K_eff_P', 0):.3g}. "
        "Для режимов PD и каскада знак определяется теми же инверсиями.")
    flag_pos = False
    if abs(e0) >= 1.0:
        early = (t <= 2.0)
        sgn = 1.0 if e0 > 0 else -1.0
        if float(np.max(sgn * e[early])) > 1.3 * abs(e0) + 1.0:
            flag_pos = True
    if res.info.get("K_eff_P", 1.0) < 0 or flag_pos:
        add("pos_fb", "ВНИМАНИЕ", "возможная ПОЛОЖИТЕЛЬНАЯ обратная связь", "",
            "Отклонение растёт или усиливается командой регулятора. Проверьте инверсии знака измерения и команды.")
    return rows


def metrics_dict(res) -> dict:
    return {r["key"]: r["value"] for r in compute_metrics(res)}


def metric_value_text(row) -> str:
    v = row["value"]
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f"{v}"
    return f"{v:.3g}"
