"""Экспорт временных рядов и параметров."""
from __future__ import annotations

import io

import pandas as pd

COLUMNS = [
    ("time_s", None, "время, с"),
    ("theta_true_deg", "theta_true", "истинный угол платформы/IMU, град"),
    ("theta_meas_deg", "theta_meas", "измеренный угол (последний пакет DMP), град"),
    ("omega_true_dps", "omega_true", "истинная угловая скорость платформы, град/с"),
    ("omega_meas_dps", "omega_meas", "измеренная угловая скорость, град/с"),
    ("omega_ref_dps", "omega_ref", "заданная скорость внутреннего контура, град/с"),
    ("control_error_deg", "e_theta", "ошибка управления по углу (0 - измеренный), град"),
    ("rate_error_dps", "e_omega", "ошибка угловой скорости (каскад), град/с"),
    ("target_pulse_us", "target_pulse", "целевой импульс (выход регулятора), мкс"),
    ("actual_pulse_us", "actual_pulse", "фактический импульс (после ограничителя), мкс"),
    ("servo_cmd_angle_deg", "cmd_angle", "заданный угол вала по фактическому импульсу, град"),
    ("servo_decoded_ref_deg", "servo_ref_angle", "декодированная серво цель (после кадра и задержки), град"),
    ("servo_angle_deg", "servo_angle", "фактический угол вала, град"),
    ("servo_omega_dps", "servo_omega", "скорость вала, град/с"),
    ("load_angle_deg", "load_angle", "угол нагрузки после люфта нагрузки, град"),
    ("base_angle_deg", "base_angle", "угол основания, град"),
    ("disturbance_deg", "base_angle", "внешнее воздействие (угол основания), град"),
    ("cmd_saturated", "cmd_sat", "команда ограничена (0/1)"),
    ("rate_limit_active", "rate_active", "ограничение скорости команды активно (0/1)"),
    ("servo_at_stop", "at_stop", "вал на упоре (0/1)"),
]


def result_to_dataframe(res) -> pd.DataFrame:
    data = {}
    for name, key, _ in COLUMNS:
        data[name] = res.t if key is None else res.s[key]
    return pd.DataFrame(data)


def csv_bytes(res) -> bytes:
    buf = io.StringIO()
    result_to_dataframe(res).to_csv(buf, index=False, float_format="%.6g")
    return buf.getvalue().encode("utf-8")
