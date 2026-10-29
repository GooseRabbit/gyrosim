"""Реестр параметров симулятора.

Каждый параметр описан один раз (ParamSpec): интерфейс, проверка и JSON строятся из этого списка.
Поле status честно показывает происхождение значения:
    user       - параметр сценария/интерфейса
    from_code  - взято из вашего скетча
    user_meas  - предварительно измерено вами (по вашему описанию проекта)
    spec       - из листингов продавцов (паспорт MG90S); независимой проверки нет, листинги местами расходятся
    calibrated - подогнано под паспортное время поворота на 60° (см. servo.calibrate_omega_nl)
    assumed    - ОЦЕНКА без источника: это допущение модели, измерьте
    unknown    - НЕИЗВЕСТНО: эффект по умолчанию выключен/равен нулю, измерьте
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

STATUS_TAGS = {
    "user": "параметр сценария",
    "from_code": "из вашего скетча",
    "user_meas": "измерено вами (по описанию проекта)",
    "spec": "из листингов продавцов; независимо не проверено, листинги местами расходятся",
    "calibrated": "подогнано под паспортное время поворота на 60 градусов",
    "assumed": "ОЦЕНКА без источника - допущение модели, измерьте",
    "unknown": "НЕИЗВЕСТНО - по умолчанию эффект выключен или равен нулю, измерьте",
}
WARN_STATUSES = ("assumed", "unknown")


class ParamError(ValueError):
    """Недопустимое значение параметра."""


@dataclass(frozen=True)
class ParamSpec:
    key: str
    group: str
    label: str
    unit: str
    kind: str                      # float | int | bool | select
    default: Any
    vmin: float | None = None
    vmax: float | None = None
    step: float | None = None
    options: tuple = ()            # для select: ((id, подпись), ...)
    help: str = ""
    status: str = "user"
    measure: str = ""
    advanced: bool = False


G_MAIN = "Основное"
G_DIST = "Внешние возмущения"
G_CTRL = "Регулятор"
G_CMD = "Команда и ограничитель (как в скетче)"
G_CAL = "Сервопривод: калибровка и геометрия"
G_DYN = "Сервопривод: динамика MG90S"
G_LOAD = "Нагрузка и механика"
G_IMU = "IMU / DMP"
G_SIM = "Симуляция"

GROUP_ORDER = [G_MAIN, G_DIST, G_CTRL, G_CMD, G_CAL, G_DYN, G_LOAD, G_IMU, G_SIM]
GROUP_OPEN = {G_MAIN: True, G_DIST: False, G_CTRL: False, G_CMD: False, G_CAL: False,
              G_DYN: False, G_LOAD: False, G_IMU: False, G_SIM: False}


def _f(key, group, label, unit, default, vmin, vmax, step, help="", status="user", measure="", advanced=False):
    return ParamSpec(key, group, label, unit, "float", float(default), float(vmin), float(vmax), float(step),
                     (), help, status, measure, advanced)


def _i(key, group, label, unit, default, vmin, vmax, step, help="", status="user", measure="", advanced=False):
    return ParamSpec(key, group, label, unit, "int", int(default), float(vmin), float(vmax), float(step),
                     (), help, status, measure, advanced)


def _b(key, group, label, default, help="", status="user", measure="", advanced=False):
    return ParamSpec(key, group, label, "", "bool", bool(default), None, None, None, (), help, status, measure, advanced)


def _s(key, group, label, default, options, help="", status="user", measure="", advanced=False):
    return ParamSpec(key, group, label, "", "select", default, None, None, None, tuple(options), help, status,
                     measure, advanced)


PERIOD_PRESETS = (
    ("5", "5 мс - исходный код (CONTROL_PERIOD_US = 5000)"),
    ("20", "20 мс - один кадр PWM"),
    ("5000", "5000 мс - текущий эксперимент (SERVO_CONTROL_PERIOD_MS = 5000)"),
    ("custom", "Своё значение (поле ниже)"),
)

PARAM_SPECS: list[ParamSpec] = [
    # ------------------------------------------------------------------ основное
    _s("ctrl_mode", G_MAIN, "Режим регулятора", "P",
       (("P", "P - текущий пропорциональный"), ("PD", "PD - демпфирование по угловой скорости"),
        ("cascade", "Каскад - угол -> скорость -> команда")),
       help="P: correction = -KP*угол. PD добавляет -KD*скорость. Каскад: внешний контур угла задаёт скорость, "
            "внутренний контур по ошибке скорости формирует команду."),
    _f("kp_us_per_deg", G_MAIN, "KP_US_PER_DEG", "мкс/град", 11.11, 0, 300, 0.5,
       help="Коэффициент пропорционального регулятора: сколько микросекунд команды на градус измеренного угла. "
            "Не путайте с безразмерным усилением петли K = KP * (град вала на мкс) * k_mech.", status="from_code"),
    _f("initial_error_deg", G_MAIN, "Начальная ошибка угла", "град", 30.0, -120, 120, 1.0,
       help="Начальное состояние платформы (не постоянное возмущение): платформа стартует отклонённой, "
            "регулятор всегда стремится вернуть измеренный угол к 0. Реализуется как начальное положение вала сервы."),
    _s("scenario", G_MAIN, "Сценарий внешнего воздействия", "initial",
       (("initial", "Только начальное отклонение"), ("step", "Ступенчатый поворот основания"),
        ("pulse", "Кратковременный импульс основания"), ("sine", "Синусоидальное возмущение"),
        ("random", "Случайное возмущение")),
       help="Воздействие на основание добавляется к начальной ошибке. Для чистого отклика на начальную ошибку "
            "выберите «Только начальное отклонение»."),
    _f("duration_s", G_MAIN, "Длительность симуляции", "с", 15.0, 1, 120, 1.0),
    _f("dt_ms", G_MAIN, "Шаг расчёта и записи", "мс", 5.0, 1, 20, 1.0,
       help="Шаг записи рядов. Внутри шага сервопривод интегрируется мелкими подшагами (<= 1 мс), "
            "поэтому результат не зависит от шага записи."),

    # ------------------------------------------------------------------ возмущения
    _f("dist_amp_deg", G_DIST, "Амплитуда возмущения основания", "град", 10.0, 0, 90, 1.0,
       help="Ступень/импульс: высота. Синус: амплитуда. Случайное: среднеквадратичное значение."),
    _f("dist_t0_s", G_DIST, "Момент начала", "с", 1.0, 0, 100, 0.1),
    _f("dist_dur_s", G_DIST, "Длительность импульса", "с", 0.5, 0.01, 50, 0.1),
    _f("dist_freq_hz", G_DIST, "Частота синуса", "Гц", 0.5, 0.01, 20, 0.05),
    _f("dist_trans_ms", G_DIST, "Время сглаживания фронтов", "мс", 50.0, 1, 2000, 10.0,
       help="Идеальная ступень имеет бесконечное ускорение; фронты сглаживаются полиномом 5-й степени."),
    _f("dist_rand_fc_hz", G_DIST, "Полоса случайного возмущения", "Гц", 1.0, 0.05, 20, 0.05),

    # ------------------------------------------------------------------ регулятор
    _f("kd_us_per_dps", G_CTRL, "KD (PD)", "мкс/(град/с)", 0.0, 0, 100, 0.05,
       help="Демпфирование по измеренной угловой скорости: correction = -(KP*угол + KD*скорость)."),
    _f("ktheta_per_s", G_CTRL, "Kθ (каскад, внешний контур)", "1/с", 2.5, 0, 50, 0.1,
       help="omega_ref = -Kθ * угол (град/с на градус)."),
    _f("kw_us_per_dps", G_CTRL, "Kω (каскад, внутренний контур)", "мкс/(град/с)", 8.0, 0, 200, 0.5,
       help="Скоростная форма: dcorrection/dt = Kω * (omega_ref - omega_измеренная). "
            "Пропорциональная форма: correction = Kω * (omega_ref - omega_измеренная); значения для скоростной формы "
            "в неё не переносятся (там это KD, мкс на град/с)."),
    _s("cascade_inner", G_CTRL, "Тип внутреннего контура", "velocity",
       (("velocity", "Скоростная форма (интегрирует ошибку скорости в команду)"),
        ("proportional", "Пропорциональный (эквивалентен PD)")),
       help="В скоростной форме ошибка скорости интегрируется в команду: это неизбежно вносит интегральное действие. "
            "Пропорциональный внутренний контур математически равен PD."),
    _f("omega_ref_max_dps", G_CTRL, "Ограничение заданной скорости", "град/с", 150.0, 5, 800, 5.0),
    _b("antiwindup", G_CTRL, "Anti-windup (ограничение состояния интегратора)", True,
       help="Если выключить, состояние интегратора скоростной формы может уходить за допустимый диапазон."),
    _b("invert_meas", G_CTRL, "Инвертировать знак измеренного угла и скорости", False,
       help="Моделирует неверный знак оси IMU."),
    _b("invert_cmd", G_CTRL, "Инвертировать направление команды сервы", False,
       help="Меняет знак коррекции. Вместе с инверсией измерения проверяет знак обратной связи."),
    _s("control_period_preset", G_CTRL, "Период обновления команды", "5", PERIOD_PRESETS,
       help="Как часто регулятор пересчитывает команду. Пресеты соответствуют версиям вашего кода."),
    _f("control_period_ms", G_CTRL, "Свой период обновления команды", "мс", 20.0, 1, 10000, 1.0,
       help="Используется только если выбран пункт «Своё значение»."),
    _b("first_update_immediate", G_CTRL, "Первое обновление команды сразу при старте", False,
       help="В скетче lastControlTime = millis() при старте, поэтому первое обновление наступает через один период."),

    # ------------------------------------------------------------------ команда и ограничитель
    _f("center_pulse_us", G_CMD, "Центральный импульс", "мкс", 1500.0, 800, 2200, 5.0, status="user_meas"),
    _f("pmin_sw_us", G_CMD, "Минимальный импульс (программный предел)", "мкс", 600.0, 400, 1500, 10.0,
       status="from_code"),
    _f("pmax_sw_us", G_CMD, "Максимальный импульс (программный предел)", "мкс", 2400.0, 1500, 2600, 10.0,
       status="from_code"),
    _f("max_correction_us", G_CMD, "Максимальная коррекция от центра", "мкс", 900.0, 10, 2000, 10.0, status="from_code"),
    _b("enable_cmd_saturation", G_CMD, "Ограничивать команду (пределы и MAX_CORRECTION)", True,
       help="Если выключить, насыщение не маскирует поведение регулятора; физическую границу задают упоры сервы."),
    _b("rate_limit_enable", G_CMD, "Программное ограничение скорости изменения команды", True,
       help="Это ограничение формирования команды в скетче (updateServoMotion), а не физическая скорость вала."),
    _f("rate_step_us", G_CMD, "Шаг команды за период ограничителя", "мкс", 10.0, 1, 500, 1.0, status="from_code"),
    _f("rate_period_ms", G_CMD, "Период ограничителя", "мс", 20.0, 1, 500, 1.0, status="from_code"),

    # ------------------------------------------------------------------ калибровка и геометрия
    _f("cal_pmin_us", G_CAL, "Калибровка: импульс нижней точки", "мкс", 500.0, 300, 1500, 10.0, status="user_meas"),
    _f("cal_pmax_us", G_CAL, "Калибровка: импульс верхней точки", "мкс", 2500.0, 1500, 2800, 10.0, status="user_meas"),
    _f("cal_amin_deg", G_CAL, "Калибровка: угол вала в нижней точке", "град", -90.0, -180, 0, 1.0, status="user_meas"),
    _f("cal_amax_deg", G_CAL, "Калибровка: угол вала в верхней точке", "град", 90.0, 0, 180, 1.0, status="user_meas"),
    _f("mech_min_deg", G_CAL, "Механический упор: нижний", "град", -90.0, -180, 0, 1.0, status="assumed",
       help="Угол в системе калибровки. Реальные упоры зависят от экземпляра.",
       measure="Медленно изменяйте импульс, пока вал не перестанет двигаться (угол по IMU перестаёт меняться)."),
    _f("mech_max_deg", G_CAL, "Механический упор: верхний", "град", 90.0, 0, 180, 1.0, status="assumed",
       measure="То же в обратную сторону."),
    _b("en_stops", G_CAL, "Учитывать механические упоры", True),
    _b("servo_enabled", G_CAL, "Серва включена", True,
       help="Если выключить, вал остаётся неподвижным при любых командах (проверка модели)."),
    _f("k_mech", G_CAL, "Коэффициент передачи угла вала в угол платформы", "", 1.0, -5, 5, 0.05, status="assumed",
       help="Угол платформы = угол основания + k_mech * угол вала. 1 означает прямое крепление платы к валу.",
       measure="Задайте угол вала командой и измерьте угол платформы по IMU при неподвижном основании."),

    # ------------------------------------------------------------------ динамика MG90S
    _s("servo_model", G_DYN, "Модель сервопривода", "electromech",
       (("electromech", "Электромеханическая (DC-мотор + насыщающий позиционный контур)"),
        ("first_order", "Упрощённая: 1-й порядок + ограничение скорости"),
        ("ideal", "Идеальная (мгновенное положение)")),
       help="Идеальная модель нужна только для сравнения; реальную серву она не описывает."),
    _s("supply", G_DYN, "Напряжение питания сервы", "4.8", (("4.8", "4.8 В"), ("6.0", "6.0 В")),
       help="Листинги дают 0.10 с/60 град и 1.8 кг*см при 4.8 В, 0.08 с/60 град и 2.2 кг*см при 6 В (без нагрузки).",
       status="spec", measure="Измерьте напряжение на выводах сервы под нагрузкой: оно просаживается."),
    _b("en_frame", G_DYN, "Серва читает импульс раз в кадр PWM", True,
       help="Любое изменение команды серва видит не раньше следующего кадра (0..20 мс ожидания)."),
    _f("pwm_frame_hz", G_DYN, "Частота кадров PWM", "Гц", 50.0, 20, 400, 1.0, status="from_code",
       help="myServo.setPeriodHertz(50)."),
    _b("en_delay", G_DYN, "Дополнительная задержка электроники/декодирования", True),
    _f("delay_ms", G_DYN, "Задержка сервы сверх кадра", "мс", 0.0, 0, 200, 1.0, status="unknown",
       help="Время от кадра с новым импульсом до начала движения. Источников для MG90S нет.",
       measure="Ступенька команды и запись угла по IMU: время до первого заметного движения минус ожидание кадра."),
    _b("en_deadband", G_DYN, "Мёртвая зона", True),
    _f("deadband_us", G_DYN, "Мёртвая зона (полная ширина)", "мкс", 5.0, 0, 50, 0.5, status="spec",
       help="В листингах MG90S указано от 1 до 5 мкс; листинги расходятся.",
       measure="Медленная пила командой 1 мкс/шаг: при каком изменении вал начинает двигаться."),
    _b("auto_omega_nl", G_DYN, "Подгонять скорость холостого хода под паспортное время 60°", True,
       help="Подбирает omega_nl так, чтобы модель проходила 60 град за паспортное время при выбранном питании "
            "(для текущих tau_a и e_sat)."),
    _f("omega_nl_dps", G_DYN, "Скорость холостого хода (если подгонка выключена)", "град/с", 1000.0, 100, 5000, 10.0,
       status="assumed", measure="Скорость на линейном участке большого шага без нагрузки."),
    _f("tau_a_ms", G_DYN, "Электромеханическая постоянная tau_a (без нагрузки)", "мс", 10.0, 1, 200, 1.0,
       status="assumed", help="Постоянная времени разгона вала: omega' = (omega_nl*u - omega)/tau_a.",
       measure="Разгон вала на большом шаге: время нарастания скорости до 63 % (по записи угла)."),
    _f("e_sat_deg", G_DYN, "Ширина насыщения усилителя e_sat", "град", 18.0, 1, 120, 1.0, status="assumed",
       help="Ошибка положения, при которой усилитель внутреннего контура насыщается.",
       measure="Подбирается по форме торможения при приближении к цели (запись угла)."),
    _f("stall_torque_nm", G_DYN, "Стартовый момент (0 = по листингу)", "Н*м", 0.0, 0, 2, 0.01, status="spec",
       help="Листинг: 1.8 кг*см при 4.8 В и 2.2 кг*см при 6 В (1 кг*см = 0.0980665 Н*м)."),
    _f("fo_tau_ms", G_DYN, "1-й порядок: постоянная времени", "мс", 30.0, 1, 500, 1.0, status="assumed",
       measure="Ступенька малой амплитуды: время до 63 % перемещения."),
    _f("fo_vmax_dps", G_DYN, "1-й порядок: предельная скорость вала", "град/с", 600.0, 50, 3000, 10.0, status="spec",
       help="Из паспортного времени 0.10 с на 60 град (4.8 В) следует средняя скорость 600 град/с без нагрузки."),
    _b("en_backlash", G_DYN, "Люфт (включает оба параметра ниже)", False),
    _f("backlash_deg", G_DYN, "Люфт редуктора ВНУТРИ контура (потенциометр на выходном валу)", "град", 0.0, 0, 10, 0.1,
       status="unknown",
       help="Мотор должен пройти люфт при реверсе, пока выходной вал не тронется. Потенциометр видит выходной вал, поэтому "
            "вал в итоге доходит до цели: люфт проявляется задержкой реверса и склонностью к «охоте». Данных нет, по умолчанию 0.",
       measure="Время реверса малой ступеньки команды по записи угла вала: избыток над временем в прямом направлении."),
    _f("backlash_load_deg", G_DYN, "Люфт нагрузки между выходным валом и платформой (мимо потенциометра)", "град", 0.0, 0, 10, 0.1,
       status="unknown",
       help="Шлиц, рычаг, трубка: потенциометр этот люфт не видит, платформа не следует за валом на его ширину при реверсе.",
       measure="Малые реверсы команды: угол платформы по IMU не следует за командой на ширину люфта."),

    # ------------------------------------------------------------------ нагрузка и механика
    _f("j_load_kgm2", G_LOAD, "Момент инерции нагрузки относительно оси вала", "кг*м^2", 0.0, 0, 0.01, 1e-6,
       status="unknown", help="Эффективный момент инерции трубки, платы IMU и креплений. Масса сама по себе его не "
                              "определяет: нужно распределение массы.",
       measure="Крутильный маятник или расчёт по CAD-модели."),
    _b("en_friction", G_LOAD, "Сухое и вязкое трение", False),
    _f("coulomb_nm", G_LOAD, "Сухое трение (момент)", "Н*м", 0.0, 0, 0.2, 0.001, status="unknown"),
    _f("viscous_nms", G_LOAD, "Вязкое демпфирование", "Н*м*с/рад", 0.0, 0, 0.2, 0.0005, status="unknown"),
    _f("grav_nm", G_LOAD, "Момент силы тяжести (амплитуда)", "Н*м", 0.0, -0.2, 0.2, 0.001, status="unknown",
       help="Момент m*g*l*sin(угол платформы). Положительное значение стремится вернуть платформу к вертикали."),
    _f("ext_torque_nm", G_LOAD, "Постоянный внешний момент сопротивления", "Н*м", 0.0, -0.2, 0.2, 0.001,
       status="unknown"),
    _b("en_inertial", G_LOAD, "Инерционный момент от ускорения основания", False,
       help="Учитывать псевдомомент -J_load*k_mech*(угловое ускорение основания) на валу."),

    # ------------------------------------------------------------------ IMU/DMP
    _f("f_dmp_hz", G_IMU, "Частота пакетов DMP (кватернион и гироскоп)", "Гц", 55.0, 10, 500, 1.0, status="assumed",
       help="По комментарию в примере SparkFun DMP работает на ~55 Гц; на вашем устройстве не измерено.",
       measure="Прирост счётчика пакетов за секунду."),
    _f("meas_delay_ms", G_IMU, "Задержка измерения угла и скорости", "мс", 0.0, 0, 500, 1.0, status="unknown",
       measure="Сравнить запись угла из DMP с независимым датчиком (например камера или энкодер) при быстром повороте."),
    _f("angle_noise_deg", G_IMU, "Шум угла (СКО)", "град", 0.0, 0, 5, 0.01, status="unknown"),
    _f("gyro_noise_dps", G_IMU, "Шум скорости (СКО)", "град/с", 0.0, 0, 20, 0.05, status="unknown"),
    _f("gyro_bias_dps", G_IMU, "Смещение нуля гироскопа", "град/с", 0.0, -20, 20, 0.05, status="unknown",
       help="Остаточное смещение после вычитания в покое."),
    _i("noise_seed", G_IMU, "Начальное значение генератора шума", "", 1, 0, 10**6, 1),
    _f("angle_filter_alpha", G_IMU, "Фильтр угла: alpha (1 = без фильтра)", "", 1.0, 0.02, 1.0, 0.02,
       help="y += alpha*(x - y) на каждом пакете. При alpha=0.2 и 55 Гц постоянная времени около 80 мс."),
    _f("gyro_filter_alpha", G_IMU, "Фильтр скорости: alpha (1 = без фильтра)", "", 1.0, 0.02, 1.0, 0.02,
       help="Тот же фильтр для скорости; в скетче GYRO_ALPHA = 0.20."),

    # ------------------------------------------------------------------ симуляция
    _f("band_deg", G_SIM, "Полоса времени установления", "град", 1.0, 0.05, 20, 0.05),
    _b("show_warnings", G_SIM, "Показывать предупреждения модели", True),
]

SPEC_BY_KEY = {s.key: s for s in PARAM_SPECS}


def defaults() -> dict:
    return {s.key: s.default for s in PARAM_SPECS}


def spec_label(spec: ParamSpec) -> str:
    mark = "⚠ " if spec.status in WARN_STATUSES else ""
    unit = f", {spec.unit}" if spec.unit else ""
    return f"{mark}{spec.label}{unit}"


def spec_help(spec: ParamSpec) -> str:
    parts = []
    if spec.help:
        parts.append(spec.help)
    parts.append(f"Статус значения: {STATUS_TAGS[spec.status]}.")
    if spec.measure:
        parts.append(f"Как измерить: {spec.measure}")
    return "\n\n".join(parts)


def validate(flat: dict) -> tuple[SimpleNamespace, list[str]]:
    """Проверяет и нормализует параметры. Возвращает (параметры, предупреждения). Бросает ParamError."""
    warnings: list[str] = []
    out: dict[str, Any] = {}
    for key in flat:
        if key not in SPEC_BY_KEY:
            warnings.append(f"Неизвестный параметр «{key}» проигнорирован.")
    for s in PARAM_SPECS:
        v = flat.get(s.key, s.default)
        if s.kind in ("float", "int"):
            try:
                x = float(v)
            except (TypeError, ValueError):
                raise ParamError(f"{s.label}: значение {v!r} не является числом")
            if not math.isfinite(x):
                raise ParamError(f"{s.label}: значение должно быть конечным числом")
            if s.vmin is not None and x < s.vmin:
                warnings.append(f"{s.label}: {x:g} меньше допустимого {s.vmin:g}, принято {s.vmin:g}")
                x = s.vmin
            if s.vmax is not None and x > s.vmax:
                warnings.append(f"{s.label}: {x:g} больше допустимого {s.vmax:g}, принято {s.vmax:g}")
                x = s.vmax
            out[s.key] = int(round(x)) if s.kind == "int" else x
        elif s.kind == "bool":
            out[s.key] = bool(v)
        else:
            ids = [o[0] for o in s.options]
            if v not in ids:
                raise ParamError(f"{s.label}: недопустимое значение {v!r}, допустимо: {ids}")
            out[s.key] = v
    p = SimpleNamespace(**out)

    # перекрёстные проверки
    if p.cal_pmax_us <= p.cal_pmin_us:
        raise ParamError("Калибровка: верхний импульс должен быть больше нижнего")
    if p.cal_amax_deg <= p.cal_amin_deg:
        raise ParamError("Калибровка: верхний угол должен быть больше нижнего")
    if p.pmax_sw_us <= p.pmin_sw_us:
        raise ParamError("Программные пределы: максимальный импульс должен быть больше минимального")
    if p.mech_max_deg <= p.mech_min_deg:
        raise ParamError("Механические упоры: верхний должен быть больше нижнего")
    if not (p.pmin_sw_us <= p.center_pulse_us <= p.pmax_sw_us):
        warnings.append("Центральный импульс лежит вне программных пределов.")
    if abs(p.k_mech) < 0.01:
        warnings.append("k_mech близок к нулю: серва практически не влияет на платформу.")
    if p.control_period_preset == "custom":
        p.control_period_s = p.control_period_ms * 1e-3
    else:
        p.control_period_s = float(p.control_period_preset) * 1e-3
    return p, warnings


def to_json(flat: dict) -> str:
    clean = {s.key: flat.get(s.key, s.default) for s in PARAM_SPECS}
    return json.dumps({"version": 1, "params": clean}, ensure_ascii=False, indent=2)


def from_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParamError(f"Файл параметров не является корректным JSON: {exc}")
    raw = data.get("params", data) if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ParamError("В файле нет словаря параметров")
    merged = defaults()
    for k, v in raw.items():
        if k in SPEC_BY_KEY:
            merged[k] = v
    validate(merged)  # бросит ParamError при ошибке
    return merged
