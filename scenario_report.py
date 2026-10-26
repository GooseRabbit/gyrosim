"""Прогон типовых сценариев и печать сводки. Запуск:  python tests/scenario_report.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gimbal_sim import metrics, params as P  # noqa: E402
from gimbal_sim import simulate as S  # noqa: E402

D = P.defaults()


def show(title, **kw):
    r = S.simulate({**D, **kw})
    m = {x["key"]: x for x in metrics.compute_metrics(r)}
    g = lambda k: metrics.metric_value_text(m[k]) if k in m else "-"
    print(f"{title:58s} max|e|={g('max_abs'):>6s}  итог={g('final_err'):>7s}  уст.={g('settle'):>6s}  "
          f"перер.={g('overshoot'):>6s}%  пересеч.={g('zero_cross'):>3s}  нас.ком.={g('sat_cmd'):>5s}%  "
          f"упор={g('sat_stop'):>5s}%  огр.PWM={g('rate_lim'):>5s}%  K={r.info['K_eff_P']:+.2f}")
    print(f"{'':58s} колебания: {m['osc']['value']}" + ("   !!! " + m["pos_fb"]["value"] if "pos_fb" in m else ""))
    return r


print("=== 1. нулевая ошибка, без возмущений ===")
show("P, KP=11.11, ошибка 0", initial_error_deg=0.0)
print("\n=== 2. малое и большое отклонение (обновление 5 мс) ===")
show("P, KP=11.11, +3°", initial_error_deg=3.0)
show("P, KP=11.11, +30°", initial_error_deg=30.0)
show("P, KP=11.11, -30°", initial_error_deg=-30.0)
show("P, KP=11.11, +60°", initial_error_deg=60.0, duration_s=20.0)
print("\n=== 3. насыщение ===")
show("P, KP=60, +60° (команда упирается в пределы)", initial_error_deg=60.0, kp_us_per_deg=60.0)
show("P, KP=11.11, +80° (вал у упора)", initial_error_deg=80.0, duration_s=20.0)
print("\n=== 4. знак обратной связи ===")
show("инверсия команды, KP=18 (K<0)", invert_cmd=True, kp_us_per_deg=18.0, initial_error_deg=10.0, duration_s=10.0)
show("инверсия команды И измерения, KP=18 (K>0)", invert_cmd=True, invert_meas=True, kp_us_per_deg=18.0, initial_error_deg=10.0)
print("\n=== 5. сравнение KP = 5 / 11.11 / 18, ограничитель ВКЛ и ВЫКЛ ===")
for lim in (True, False):
    for kp in (5.0, 11.11, 18.0):
        show(f"KP={kp:g}, ограничитель {'ВКЛ' if lim else 'ВЫКЛ'}", kp_us_per_deg=kp, rate_limit_enable=lim, initial_error_deg=30.0)
print("\n=== 6. период обновления: исходный 5 мс и текущий эксперимент 5 с ===")
for kp in (6.0, 11.11, 15.0):
    show(f"KP={kp:g}, обновление 5000 мс", kp_us_per_deg=kp, control_period_preset="5000", duration_s=120.0, initial_error_deg=10.0)
print("\n=== 7. каскад и PD (обновление 5 мс) ===")
show("PD, KP=11.11, KD=0.3", ctrl_mode="PD", kd_us_per_dps=0.3)
show("Каскад, скоростная форма Kθ=2.5 Kω=8", ctrl_mode="cascade")
print("\n=== 8. реакция на возмущение основания: ступень 10°, остаток по формуле 1/(1+K) ===")
for kp in (11.11, 18.0):
    r = show(f"P, KP={kp:g}, ступень основания 10°", scenario="step", dist_amp_deg=10.0, initial_error_deg=0.0, kp_us_per_deg=kp,
             duration_s=10.0)
    print(f"{'':58s} остаток в конце {r.s['theta_true'][-1]:+.2f}°, формула 10/(1+K) = {10 / (1 + r.info['K_eff_P']):+.2f}°")
print("\n=== 9. Что нужно для реальной раскачки при быстром обновлении? Неизвестные задержки (ПРИМЕР, не измерено) ===")
for kp in (11.11, 18.0, 30.0):
    for dly in ((0.0, 0.0), (40.0, 10.0), (80.0, 20.0)):
        r = S.simulate({**D, "kp_us_per_deg": kp, "meas_delay_ms": dly[0], "delay_ms": dly[1], "duration_s": 15.0})
        m = {x["key"]: x for x in metrics.compute_metrics(r)}
        print(f"  KP={kp:5.2f} (K={r.info['K_eff_P']:.2f}) задержка IMU {dly[0]:3.0f} мс, сервы {dly[1]:3.0f} мс -> {m['osc']['value']}")
