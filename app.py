"""Интерактивный математический симулятор гиростабилизатора ESP32 + ICM-20948 + MG90S.

Запуск:   streamlit run app.py
Модель:   gimbal_sim/ (параметры, сервопривод, IMU, регуляторы, замкнутый контур, метрики)
ВНИМАНИЕ: результаты зависят от точности модели. Динамические параметры MG90S не публикуются производителем:
          всё, что помечено знаком предупреждения, нужно определить измерением на вашем устройстве.
"""
from __future__ import annotations

import inspect
import io
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gimbal_sim import export as EX  # noqa: E402
from gimbal_sim import identification as ID  # noqa: E402
from gimbal_sim import metrics as MT  # noqa: E402
from gimbal_sim import params as P  # noqa: E402
from gimbal_sim import plots as PL  # noqa: E402
from gimbal_sim import servo as SV  # noqa: E402
from gimbal_sim import servo_tests as ST  # noqa: E402
from gimbal_sim import simulate as SIM  # noqa: E402

st.set_page_config(page_title="Симулятор гиростабилизатора", page_icon="🎛️", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.4rem;}
.hero {border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;
       background: linear-gradient(120deg, #14213d 0%, #1b2a49 55%, #0f3d4c 100%); border: 1px solid #2b3a5c;}
.hero h1 {margin: 0 0 4px 0; font-size: 1.7rem;}
.hero p {margin: 0; color: #b9c3d9; font-size: 0.95rem;}
.chip {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; margin-top: 10px;}
.chip.ok {background: #12351f; color: #6ee7a0; border: 1px solid #1f6b3b;}
.chip.stale {background: #3d2d0b; color: #ffd166; border: 1px solid #7a5a12;}
.chip.none {background: #1f2937; color: #cbd5e1; border: 1px solid #374151;}
</style>
"""


# --------------------------------------------------------------------------- совместимость версий Streamlit
def _wide_kw(fn) -> dict:
    """Растянуть элемент по ширине: use_container_width в старых версиях, width='stretch' в новых."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "width" in params:
        return {"width": "stretch"}
    return {}


def _has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def show_chart(fig, key: str):
    kw = _wide_kw(st.plotly_chart)
    if _has_param(st.plotly_chart, "key"):
        kw["key"] = key
    st.plotly_chart(fig, **kw)


def show_df(df: pd.DataFrame):
    st.dataframe(df, **_wide_kw(st.dataframe))


def wide_button(container, label: str, **kw):
    kw.update(_wide_kw(container.button))
    return container.button(label, **kw)


# --------------------------------------------------------------------------- состояние
EXTRA_STATE = {
    "result": None, "result_sig": None, "compare": None, "compare_sig": None, "error": None, "msg": None,
    "servo_tests": None, "autotest_report": None, "ident_sets": [], "ident_fit": None, "kp_list_text": "5, 11.11, 18",
    "export_cache": None,
}


def init_state():
    for spec in P.PARAM_SPECS:
        if spec.key not in st.session_state:
            st.session_state[spec.key] = spec.default
    for k, v in EXTRA_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = list(v) if isinstance(v, list) else v


def current_flat() -> dict:
    return {s.key: st.session_state[s.key] for s in P.PARAM_SPECS}


def signature(flat: dict) -> str:
    return "|".join(f"{k}={flat[k]!r}" for k in sorted(flat))


def reset_params():
    for spec in P.PARAM_SPECS:
        st.session_state[spec.key] = spec.default
    for k in ("result", "result_sig", "compare", "compare_sig", "error", "servo_tests", "ident_fit", "export_cache"):
        st.session_state[k] = None
    st.session_state["msg"] = ("ok", "Настройки сброшены к значениям по умолчанию.")


def load_params_cb():
    up = st.session_state.get("params_upload")
    if up is None:
        return
    try:
        flat = P.from_json(up.getvalue().decode("utf-8"))
        p, _ = P.validate(flat)
        for spec in P.PARAM_SPECS:
            st.session_state[spec.key] = getattr(p, spec.key)
        st.session_state["msg"] = ("ok", "Параметры загружены из файла. Нажмите «Запустить симуляцию».")
    except Exception as exc:  # noqa: BLE001
        st.session_state["msg"] = ("err", f"Не удалось загрузить параметры: {exc}")


def apply_fit_cb():
    fit = st.session_state.get("ident_fit")
    if not fit:
        return
    for k, v in fit["params"].items():
        st.session_state[k] = float(v)
    st.session_state["servo_model"] = "electromech"
    st.session_state["auto_omega_nl"] = False
    st.session_state["msg"] = ("ok", "Подобранные параметры сервы применены (подгонка omega_nl под паспорт отключена).")


# --------------------------------------------------------------------------- боковая панель
DISABLE_RULES = {
    "control_period_ms": lambda s: s["control_period_preset"] != "custom",
    "kd_us_per_dps": lambda s: s["ctrl_mode"] != "PD",
    "ktheta_per_s": lambda s: s["ctrl_mode"] != "cascade",
    "kw_us_per_dps": lambda s: s["ctrl_mode"] != "cascade",
    "cascade_inner": lambda s: s["ctrl_mode"] != "cascade",
    "omega_ref_max_dps": lambda s: s["ctrl_mode"] != "cascade",
    "antiwindup": lambda s: s["ctrl_mode"] != "cascade",
    "omega_nl_dps": lambda s: bool(s["auto_omega_nl"]),
    "dist_amp_deg": lambda s: s["scenario"] == "initial",
    "dist_t0_s": lambda s: s["scenario"] in ("initial", "random"),
    "dist_dur_s": lambda s: s["scenario"] != "pulse",
    "dist_freq_hz": lambda s: s["scenario"] != "sine",
    "dist_trans_ms": lambda s: s["scenario"] not in ("step", "pulse"),
    "dist_rand_fc_hz": lambda s: s["scenario"] != "random",
}


def render_param(spec: P.ParamSpec, box):
    label = P.spec_label(spec)
    help_text = P.spec_help(spec)
    dis = bool(DISABLE_RULES[spec.key](st.session_state)) if spec.key in DISABLE_RULES else False
    if spec.kind == "float":
        box.number_input(label, min_value=spec.vmin, max_value=spec.vmax, step=spec.step, key=spec.key,
                         help=help_text, format="%g", disabled=dis)
    elif spec.kind == "int":
        box.number_input(label, min_value=int(spec.vmin), max_value=int(spec.vmax), step=int(spec.step), key=spec.key,
                         help=help_text, disabled=dis)
    elif spec.kind == "bool":
        box.checkbox(label, key=spec.key, help=help_text, disabled=dis)
    else:
        names = dict(spec.options)
        box.selectbox(label, options=[o[0] for o in spec.options], format_func=lambda x, m=names: m[x], key=spec.key,
                      help=help_text, disabled=dis)


def render_sidebar():
    sb = st.sidebar
    sb.markdown("### Параметры")
    sb.caption("Знак ⚠ - значение не измерено на вашей серве или неизвестно; подсказка у поля объясняет, как его измерить.")
    for group in P.GROUP_ORDER:
        specs = [s for s in P.PARAM_SPECS if s.group == group]
        with sb.expander(group, expanded=P.GROUP_OPEN.get(group, False)):
            for spec in specs:
                render_param(spec, st)
    with sb.expander("Сохранить / загрузить набор параметров", expanded=False):
        st.download_button("Сохранить параметры (JSON)", data=P.to_json(current_flat()).encode("utf-8"),
                           file_name="gimbal_params.json", mime="application/json")
        st.file_uploader("Загрузить параметры (JSON)", type=["json"], key="params_upload", on_change=load_params_cb)


# --------------------------------------------------------------------------- расчёты
def run_simulation(flat: dict):
    st.session_state["error"] = None
    try:
        with st.spinner("Расчёт..."):
            res = SIM.simulate(flat)
        st.session_state["result"] = res
        st.session_state["result_sig"] = signature(flat)
    except (P.ParamError, SIM.SimulationError) as exc:
        st.session_state["error"] = str(exc)


def parse_kp_list(text: str) -> list[float]:
    text = text.strip()
    if ";" in text:
        toks = [t.strip().replace(",", ".") for t in text.split(";")]
    else:
        toks = [t for t in re.split(r"[,\s]+", text) if t]
    vals = []
    for t in toks:
        if not t:
            continue
        v = float(t)
        if not (0.0 <= v <= 300.0):
            raise ValueError(f"KP = {v:g} вне диапазона 0..300")
        vals.append(v)
    if not vals:
        raise ValueError("Список KP пуст")
    if len(vals) > 8:
        raise ValueError("Не более 8 значений KP")
    return vals


def run_compare(flat: dict):
    st.session_state["error"] = None
    try:
        kps = parse_kp_list(st.session_state["kp_list_text"])
        with st.spinner(f"Расчёт {len(kps)} вариантов..."):
            results = SIM.simulate_kp_list(flat, kps)
        st.session_state["compare"] = {"kps": kps, "results": results}
        st.session_state["compare_sig"] = signature(flat)
    except (P.ParamError, SIM.SimulationError, ValueError) as exc:
        st.session_state["error"] = f"Сравнение: {exc}"


def metrics_table(res) -> pd.DataFrame:
    rows = MT.compute_metrics(res)
    return pd.DataFrame({"Показатель": [r["label"] for r in rows],
                         "Значение": [MT.metric_value_text(r) for r in rows],
                         "Ед.": [r["unit"] for r in rows],
                         "Что это значит": [r["help"] for r in rows]})


def controller_formulas(p) -> list[tuple[str, str, str]]:
    """(заголовок, LaTeX, пояснение с текущими параметрами)."""
    out = [("Кинематика замкнутого контура",
            r"\theta_p=\theta_b+k_{mech}\,\theta_s,\qquad \theta_{IMU}=\theta_p",
            f"θp - угол платформы (и IMU), θb - угол основания, θs - ФАКТИЧЕСКИЙ угол вала, k_mech = {p.k_mech:g}. "
            "IMU сидит на платформе, поэтому движение вала меняет измерение, а измерение меняет команду.")]
    if p.ctrl_mode == "P":
        out.append(("P (текущий регулятор скетча)",
                    r"c=-K_P\,\hat\theta,\qquad u_{target}=u_0+s_{cmd}\,c",
                    f"c - коррекция импульса, мкс; K_P = {p.kp_us_per_deg:g} мкс/град; u0 = {p.center_pulse_us:g} мкс; "
                    f"θ̂ - последний пакет угла DMP, град. Команда ограничивается [{p.pmin_sw_us:g}; {p.pmax_sw_us:g}] мкс и "
                    f"|c| ≤ {p.max_correction_us:g} мкс."))
    elif p.ctrl_mode == "PD":
        out.append(("PD (демпфирование по угловой скорости гироскопа)",
                    r"c=-\left(K_P\,\hat\theta+K_D\,\hat\omega\right)",
                    f"K_P = {p.kp_us_per_deg:g} мкс/град; K_D = {p.kd_us_per_dps:g} мкс/(град/с); ω̂ - измеренная скорость, град/с."))
    else:
        out.append(("Каскад: внешний контур (угол -> заданная скорость)",
                    r"\omega_{ref}=\mathrm{clip}\left(-K_\theta\,\hat\theta,\ \pm\omega_{max}\right)",
                    f"Kθ = {p.ktheta_per_s:g} 1/с; ω_max = {p.omega_ref_max_dps:g} град/с."))
        if p.cascade_inner == "velocity":
            out.append(("Каскад: внутренний контур, скоростная форма",
                        r"\dot c=K_\omega\left(\omega_{ref}-\hat\omega\right)\ \Rightarrow\ c=K_\omega\!\int\!\left(\omega_{ref}-\hat\omega\right)dt",
                        f"Kω = {p.kw_us_per_dps:g} мкс/(град/с). Здесь интегрируется ошибка скорости: это даёт интегральное "
                        f"действие (при идеальных датчиках эквивалентно ПИ: P-часть Kω, I-часть Kω·Kθ = {p.kw_us_per_dps * p.ktheta_per_s:g} мкс/(град·с)). "
                        "Anti-windup: " + ("включён" if p.antiwindup else "выключен") + "."))
        else:
            out.append(("Каскад: внутренний контур, пропорциональный",
                        r"c=K_\omega\left(\omega_{ref}-\hat\omega\right)=-K_\omega K_\theta\,\hat\theta-K_\omega\,\hat\omega",
                        f"Это PD с K_P = Kω·Kθ = {p.kw_us_per_dps * p.ktheta_per_s:g} мкс/град и K_D = Kω = {p.kw_us_per_dps:g} мкс/(град/с)."))
    out.append(("Усиление петли (статическое, P)",
                r"K=K_P\cdot\frac{\Delta\theta_{cal}}{\Delta u_{cal}}\cdot k_{mech}\cdot s_{cmd}\,s_{meas}",
                "K > 0 - отрицательная обратная связь; остаток наклона основания 1/(1+K). При очень медленном обновлении команды "
                "(период >> время установления сервы) для устойчивости нужно K < 1."))
    out.append(("Сервопривод (электромеханическая модель)",
                r"e=\theta_{ref}-\theta_{out},\ \ u=\mathrm{clip}\!\left(\tfrac{e}{e_{sat}},\pm1\right),\ \ T_m=T_{stall}\left(u-\tfrac{\omega}{\omega_{nl}}\right),\ \ J\dot\omega=T_m-c_v\omega-T_c\,\mathrm{sgn}\,\omega-T_{grav}-T_{ext}",
                "Потенциометр измеряет ВЫХОДНОЙ вал. Дополнительно: кадр PWM, задержка, мёртвая зона, люфт, упоры."))
    return out


# --------------------------------------------------------------------------- вкладки
def tab_simulation(flat: dict):
    res = st.session_state["result"]
    if st.session_state["error"]:
        st.error(st.session_state["error"])
    if res is None:
        st.info("Задайте параметры слева и нажмите «Запустить симуляцию». Начальная ошибка - начальное состояние платформы; "
                "регулятор всегда возвращает измеренный угол к 0°.")
        return
    p = res.params
    st.caption(f"График построен с параметрами: режим {p.ctrl_mode}, KP = {p.kp_us_per_deg:g} мкс/град, начальная ошибка "
               f"{p.initial_error_deg:g}°, сценарий «{p.scenario}», период обновления {p.control_period_s * 1e3:g} мс, "
               f"серва «{p.servo_model}», ограничитель PWM {'вкл' if p.rate_limit_enable else 'выкл'}.")
    if p.show_warnings:
        for w in res.warnings:
            st.warning(w)
    rows = {r["key"]: r for r in MT.compute_metrics(res)}
    cols = st.columns(6)
    for col, key in zip(cols, ("max_abs", "final_err", "rmse", "settle", "overshoot", "zero_cross")):
        r = rows[key]
        col.metric(r["label"], MT.metric_value_text(r) + (f" {r['unit']}" if r["unit"] and r["value"] is not None else ""))
    osc = rows["osc"]
    (st.error if ("нарастающие" in osc["value"] or "предельный" in osc["value"]) else st.info)(f"Колебания: {osc['value']}. {osc['help']}")
    if "pos_fb" in rows:
        st.error("ВНИМАНИЕ: " + rows["pos_fb"]["value"] + ". " + rows["pos_fb"]["help"])
    show_chart(PL.build_main_figure(res), "main_chart")
    with st.expander("Все показатели и пояснения", expanded=False):
        show_df(metrics_table(res))
    with st.expander("Формулы и единицы выбранного режима", expanded=False):
        for title, latex, note in controller_formulas(p):
            st.markdown(f"**{title}**")
            st.latex(latex)
            st.caption(note)
    info = res.info
    st.caption(f"Расчёт: подшаг {info['substep_s'] * 1e3:.3g} мс, пакетов DMP {info['packets']}, обновлений команды "
               f"{info['controller_updates']}. Статика P: K = {info['K_eff_P']:+.3g}, ожидаемый остаток 1/(1+K) = "
               f"{info['residual_fraction_P']:.3g}; обратная связь: {info['feedback']}.")


def tab_compare(flat: dict):
    st.text_input("Список KP_US_PER_DEG (через запятую или пробел; при десятичной запятой используйте «;»)", key="kp_list_text")
    if wide_button(st, "Сравнить коэффициенты (эта вкладка)", key="compare_local"):
        run_compare(flat)
    comp = st.session_state["compare"]
    if st.session_state["error"] and comp is None:
        st.error(st.session_state["error"])
    if comp is None:
        st.info("Для каждого KP используются одинаковые начальная ошибка, возмущения и параметры сервы.")
        return
    results, kps = comp["results"], comp["kps"]
    labels = [f"KP = {k:g}" for k in kps]
    show_chart(PL.build_compare_figure(results, labels), "compare_chart")
    table = []
    for k, r in zip(kps, results):
        m = {x["key"]: x for x in MT.compute_metrics(r)}
        table.append({"KP": k, "K петли": round(r.info["K_eff_P"], 3),
                      "макс. |ошибка|, °": MT.metric_value_text(m["max_abs"]), "остаток, °": MT.metric_value_text(m["final_err"]),
                      "установление, с": MT.metric_value_text(m["settle"]), "перерегулирование, %": MT.metric_value_text(m["overshoot"]),
                      "пересечений нуля": MT.metric_value_text(m["zero_cross"]), "колебания": m["osc"]["value"],
                      "насыщение команды, %": MT.metric_value_text(m["sat_cmd"]), "упор, %": MT.metric_value_text(m["sat_stop"]),
                      "огр. PWM, %": MT.metric_value_text(m["rate_lim"])})
    show_df(pd.DataFrame(table))
    if results[0].params.rate_limit_enable:
        st.caption("При включённом программном ограничителе (10 мкс за 20 мс) первые секунды отклика определяются ограничителем, "
                   "а не KP: отключите его в разделе «Команда и ограничитель», чтобы увидеть влияние KP.")


def tab_servo(flat: dict):
    st.caption("Испытания сервопривода отдельно от контура (неподвижное основание). Для каждого показаны целевой и фактический "
               "импульс, заданный и фактический угол вала, а также автоматические проверки.")
    if wide_button(st, "Запустить испытания сервопривода", key="run_servo_tests"):
        try:
            p, _ = P.validate(flat)
            with st.spinner("Испытания..."):
                st.session_state["servo_tests"] = ST.run_suite(p)
        except (P.ParamError, SIM.SimulationError) as exc:
            st.error(str(exc))
    tests = st.session_state["servo_tests"]
    if not tests:
        return
    try:
        p, _ = P.validate(flat)
        sm = SV.electromech_small_signal(p)
        t60 = SV.time_to_reach(P.validate({**flat, "en_frame": False, "en_delay": False, "en_deadband": False})[0], 60.0, 1.0)
        sup = SV.SUPPLY_SPEC[p.supply]
        st.info(f"Проверка по паспорту ({p.supply} В): время 60° без нагрузки в листингах {sup['t60_s']:.2f} с; модель (только движение) "
                f"{t60:.3f} с. Внутренний контур модели: omega_nl = {sm['omega_nl_dps']:.0f} град/с, ωn = {sm['omega_n_rad_s']:.0f} рад/с, "
                f"ζ = {sm['zeta']:.2f}.")
    except Exception:  # noqa: BLE001
        pass
    names = [t["name"] for t in tests]
    sel = st.selectbox("Испытание", names, key="servo_test_sel")
    t = tests[names.index(sel)] if sel in names else tests[0]
    st.caption(t["desc"])
    show_chart(PL.build_servo_test_figure(t), "servo_test_chart")
    for txt, ok in t["checks"]:
        (st.success if ok else st.error)(("ПРОЙДЕНО: " if ok else "НЕ ПРОЙДЕНО: ") + txt)


def _demo_dataset(flat: dict):
    """Синтетические «измерения» из модели с известными параметрами и шумом: только для демонстрации функции."""
    f = {**flat, "servo_model": "electromech", "auto_omega_nl": False, "tau_a_ms": 14.0, "e_sat_deg": 25.0, "omega_nl_dps": 650.0,
         "en_frame": False, "en_delay": True, "delay_ms": 9.0, "en_deadband": False}
    p, _ = P.validate(f)
    cal = SV.Calibration(p)
    rng = np.random.default_rng(5)
    sets = []
    for amp in (60.0, 8.0):
        p1, p0 = cal.angle_to_pulse(amp), cal.p_center
        r = SV.servo_response(p, lambda t, a=p1, b=p0: a if t >= 0.05 else b, 0.6, 1e-3)
        t = np.arange(0, 0.6, 0.004)
        a = np.interp(t, r["t"], r["angle"]) + rng.normal(0, 0.15, t.size)
        sets.append({"label": f"ДЕМО (синтетика) ступенька {amp:g}°", "ds": ID.StepDataset(t, a, p0, p1, 0.05)})
    return sets


def tab_ident(flat: dict):
    st.markdown("**Сравнение модели сервопривода с вашими измерениями.** Загрузите запись ступенчатого отклика "
                "(CSV: время в секундах; угол в градусах) и укажите импульсы до и после ступеньки.")
    st.warning("Пока нет реальных измерений, параметры сервы в модели остаются предварительными. Демонстрационные данные "
               "ниже синтетические и нужны только чтобы показать работу подбора.")
    c1, c2, c3 = st.columns(3)
    pf = c1.number_input("Импульс до ступеньки, мкс", min_value=400.0, max_value=2600.0, value=1500.0, step=10.0, key="id_pf")
    pt = c2.number_input("Импульс после ступеньки, мкс", min_value=400.0, max_value=2600.0, value=2167.0, step=10.0, key="id_pt")
    ts = c3.number_input("Момент подачи ступеньки, с", min_value=0.0, max_value=10.0, value=0.05, step=0.01, format="%g", key="id_ts")
    up = st.file_uploader("CSV с измерением (время; угол)", type=["csv", "txt"], key="id_csv")
    b1, b2, b3 = st.columns(3)
    if wide_button(b1, "Добавить измерение в набор", key="id_add"):
        if up is None:
            st.error("Сначала выберите CSV-файл.")
        else:
            try:
                t, a = ID.load_measurement_csv(up.getvalue().decode("utf-8", errors="replace"))
                st.session_state["ident_sets"].append({"label": f"{up.name}: {pf:g} -> {pt:g} мкс",
                                                       "ds": ID.StepDataset(t, a, pf, pt, ts)})
            except ValueError as exc:
                st.error(str(exc))
    if wide_button(b2, "Загрузить демонстрационные данные", key="id_demo"):
        st.session_state["ident_sets"] = _demo_dataset(flat)
        st.session_state["ident_fit"] = None
    if wide_button(b3, "Очистить набор", key="id_clear"):
        st.session_state["ident_sets"] = []
        st.session_state["ident_fit"] = None
    sets = st.session_state["ident_sets"]
    if not sets:
        st.info("Набор измерений пуст.")
    else:
        st.write("В наборе: " + "; ".join(s["label"] for s in sets))
        datasets = [s["ds"] for s in sets]
        labels = [s["label"] for s in sets]
        try:
            show_chart(PL.build_identification_figure(ID.compare(flat, datasets), labels), "id_cmp_chart")
        except (P.ParamError, ValueError) as exc:
            st.error(str(exc))
        if wide_button(st, "Подобрать параметры (tau_a, e_sat, omega_nl, задержка)", key="id_fit_btn"):
            try:
                with st.spinner("Подбор..."):
                    st.session_state["ident_fit"] = ID.fit_servo(flat, datasets)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Подбор не удался: {exc}")
        fit = st.session_state["ident_fit"]
        if fit:
            st.success(f"Ошибка RMSE: {fit['rmse_before']:.3g}° -> {fit['rmse_after']:.3g}°. Подобрано: "
                       + ", ".join(f"{k} = {v:.4g}" for k, v in fit["params"].items()))
            st.caption("Параметры tau_a, e_sat и omega_nl в данных частично взаимозависимы: проверьте, что набор содержит и "
                       "большую, и малую ступеньки. Кадры PWM при подборе отключены, их эффект включает задержка.")
            try:
                show_chart(PL.build_identification_figure(ID.compare(fit["flat"], datasets), labels), "id_fit_chart")
            except (P.ParamError, ValueError) as exc:
                st.error(str(exc))
            st.button("Применить найденные параметры к симулятору", on_click=apply_fit_cb, key="id_apply")
    st.markdown("---")
    st.markdown("**Сводка измеренных характеристик vs модель** (оставьте 0, если не измеряли)")
    d1, d2, d3, d4, d5 = st.columns(5)
    m_start = d1.number_input("Задержка начала движения, с", min_value=0.0, value=0.0, step=0.005, format="%g", key="m_start")
    m_reach = d2.number_input("Время достижения 95 %, с", min_value=0.0, value=0.0, step=0.005, format="%g", key="m_reach")
    m_settle = d3.number_input("Время установления 2 %, с", min_value=0.0, value=0.0, step=0.005, format="%g", key="m_settle")
    m_vmax = d4.number_input("Макс. скорость, град/с", min_value=0.0, value=0.0, step=10.0, format="%g", key="m_vmax")
    m_over = d5.number_input("Перерегулирование, %", min_value=0.0, value=0.0, step=1.0, format="%g", key="m_over")
    try:
        ch = ID.model_step_characteristics(flat, pf, pt)
    except (P.ParamError, ValueError) as exc:
        st.error(str(exc))
        return
    rows = []
    for name, meas, key, unit in (("Задержка начала движения", m_start, "start_delay_s", "с"),
                                  ("Время достижения 95 %", m_reach, "reach_95_s", "с"),
                                  ("Время установления 2 %", m_settle, "settle_2pct_s", "с"),
                                  ("Максимальная скорость", m_vmax, "max_speed_dps", "град/с"),
                                  ("Перерегулирование", m_over, "overshoot_pct", "%")):
        mv = ch.get(key)
        rows.append({"Характеристика": name, "Модель": "—" if mv is None else f"{mv:.4g}",
                     "Измерено": "—" if meas == 0 else f"{meas:.4g}", "Ед.": unit,
                     "Разница": "—" if (meas == 0 or mv is None) else f"{mv - meas:+.3g}"})
    show_df(pd.DataFrame(rows))


def tab_checks():
    st.caption("Автоматические проверки математической модели и контроллера (папка tests/). Они не требуют интерфейса.")
    if wide_button(st, "Запустить автопроверки", key="run_autotests"):
        with st.spinner("Выполняются проверки..."):
            loader = unittest.defaultTestLoader
            suite = loader.discover(os.path.join(ROOT, "tests"), pattern="test_*.py")
            stream = io.StringIO()
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            st.session_state["autotest_report"] = (result.testsRun, len(result.failures), len(result.errors), stream.getvalue())
    rep = st.session_state["autotest_report"]
    if rep:
        n, nf, ne, text = rep
        (st.success if (nf == 0 and ne == 0) else st.error)(f"Выполнено проверок: {n}; провалено: {nf}; ошибок: {ne}.")
        st.code(text)


def tab_model():
    st.markdown(
        "### Что моделируется\n"
        "Замкнутый контур по одной оси: **основание -> серва -> платформа с IMU -> DMP-пакеты -> регулятор -> команда PWM -> "
        "ограничитель -> серва**. Серва поворачивает платформу вместе с IMU, поэтому ориентация IMU определяется "
        "ФАКТИЧЕСКИМ положением вала, а не заданным импульсом.\n\n"
        "**Упрощения.** Одномерная модель угла Twist (ось вала предполагается выбранной верно, swing = 0). Угол Twist из "
        "кватерниона: twist = 2·atan2(n·v, w) для q_rel = conj(q_ref)·q_now; в симуляторе этот шаг не воспроизводится. "
        "Жёсткая платформа без упругости; люфт и трение - простейшие модели; расход тока и просадка питания не моделируются.\n\n"
        "**Начальная ошибка** - начальное положение вала (платформа стартует отклонённой), а не постоянное возмущение. "
        "Цель регулятора всегда 0°.")
    st.markdown("### Данные о MG90S и их достоверность")
    table = pd.DataFrame([
        {"Характеристика": "Время поворота на 60° без нагрузки", "Значение": "0.10 с при 4.8 В; 0.08 с при 6 В",
         "Статус": "листинги продавцов совпадают между собой; независимо не проверено; зависит от напряжения и нагрузки"},
        {"Характеристика": "Стартовый момент", "Значение": "1.8 кг·см при 4.8 В; 2.2 кг·см при 6 В", "Статус": "листинги; зависит от напряжения"},
        {"Характеристика": "Ход вала", "Значение": "около 180° (±90°)", "Статус": "листинги; реальный диапазон зависит от экземпляра"},
        {"Характеристика": "Мёртвая зона", "Значение": "1-5 мкс", "Статус": "листинги расходятся"},
        {"Характеристика": "Аналоговая или цифровая", "Значение": "—", "Статус": "листинги противоречат друг другу"},
        {"Характеристика": "Структура привода", "Значение": "DC-мотор, редуктор, потенциометр на выходном валу, усилитель с насыщением",
         "Статус": "подтверждено описанием разборки SG90 (родственная серва)"},
        {"Характеристика": "Электромеханическая постоянная, коэффициент внутреннего контура, задержка, люфт, трение, инерция",
         "Значение": "нет данных", "Статус": "производитель не публикует; определять измерением (вкладка «Идентификация»)"},
    ])
    show_df(table)
    st.markdown("### Параметры, которые требуют внимания")
    rows = []
    for s in P.PARAM_SPECS:
        if s.status in ("assumed", "unknown", "calibrated", "spec"):
            rows.append({"Группа": s.group, "Параметр": s.label, "По умолчанию": str(s.default),
                         "Статус": P.STATUS_TAGS[s.status], "Как измерить": s.measure or "—"})
    show_df(pd.DataFrame(rows))
    st.warning("Результаты симулятора зависят от точности модели и должны сверяться с экспериментами на реальном устройстве. "
               "Предварительная модель не является идентичной реальной MG90S.")


def tab_export():
    res = st.session_state["result"]
    if res is None:
        st.info("Сначала выполните симуляцию.")
        return
    st.download_button("Скачать временные ряды (CSV)", data=EX.csv_bytes(res), file_name="gimbal_sim_result.csv", mime="text/csv")
    st.download_button("Сохранить набор параметров (JSON)", data=P.to_json(current_flat()).encode("utf-8"),
                       file_name="gimbal_params.json", mime="application/json")
    st.caption("PNG и HTML готовятся по кнопке, чтобы не замедлять интерфейс при каждом изменении параметров.")
    if wide_button(st, "Подготовить графики для скачивания (PNG, HTML)", key="prep_export"):
        fig = PL.build_main_figure(res)
        html = PL.figure_html_bytes(fig)
        try:
            png, note = PL.figure_png_bytes(fig), "PNG из Plotly (kaleido)."
        except RuntimeError:
            png, note = PL.matplotlib_png(res), "PNG построен через matplotlib (для Plotly-PNG нужен пакет kaleido)."
        st.session_state["export_cache"] = {"sig": st.session_state["result_sig"], "html": html, "png": png, "note": note}
    cache = st.session_state.get("export_cache")
    if cache and cache["sig"] == st.session_state["result_sig"]:
        st.download_button("Скачать график (HTML, работает без интернета)", data=cache["html"],
                           file_name="gimbal_sim_plot.html", mime="text/html")
        st.download_button("Скачать график (PNG)", data=cache["png"], file_name="gimbal_sim_plot.png", mime="image/png")
        st.caption(cache["note"])


# --------------------------------------------------------------------------- страница
def main():
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()
    flat = current_flat()
    sig = signature(flat)

    res = st.session_state["result"]
    if res is None:
        chip = '<span class="chip none">симуляция ещё не запускалась</span>'
    elif st.session_state["result_sig"] == sig:
        chip = '<span class="chip ok">результат соответствует текущим параметрам</span>'
    else:
        chip = '<span class="chip stale">параметры изменены после запуска - график показывает ПРЕДЫДУЩИЙ результат</span>'
    st.markdown(f'<div class="hero"><h1>Симулятор гиростабилизатора</h1>'
                f'<p>ESP32 + ICM-20948 (DMP) + MG90S: замкнутая система «платформа с IMU - серва - регулятор». '
                f'Цель регулятора всегда 0°; изменяется только начальная ошибка и внешние воздействия.</p>{chip}</div>',
                unsafe_allow_html=True)
    st.info("Результаты зависят от точности модели. Параметры с ⚠ не измерены на вашей серве - сверяйте расчёт с экспериментом.")

    msg = st.session_state.get("msg")
    if msg:
        (st.success if msg[0] == "ok" else st.error)(msg[1])
        st.session_state["msg"] = None

    c1, c2, c3, c4 = st.columns(4)
    run_clicked = wide_button(c1, "▶ Запустить симуляцию", type="primary", key="btn_run")
    restart_clicked = wide_button(c2, "↻ Перезапустить с текущими параметрами", key="btn_restart")
    c3.button("Сбросить настройки", on_click=reset_params, key="btn_reset", **_wide_kw(c3.button))
    compare_clicked = wide_button(c4, "Сравнить коэффициенты", key="btn_compare")

    if run_clicked or restart_clicked:
        if restart_clicked:
            st.session_state["result"] = None
            st.session_state["result_sig"] = None
        run_simulation(flat)
    if compare_clicked:
        run_compare(flat)

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Симуляция", "Сравнение KP", "Сервопривод", "Идентификация MG90S",
                                          "Проверки", "Модель и данные", "Экспорт"])
    with t1:
        tab_simulation(flat)
    with t2:
        tab_compare(flat)
    with t3:
        tab_servo(flat)
    with t4:
        tab_ident(flat)
    with t5:
        tab_checks()
    with t6:
        tab_model()
    with t7:
        tab_export()


main()
