"""Модель измерения IMU/DMP.

DMP выдаёт ПАКЕТЫ с частотой f_dmp: оценку угла Twist (из кватерниона) и угловую скорость гироскопа. Между пакетами
регулятор видит последнее значение (ZOH). Модель добавляет: задержку, шум, смещение нуля гироскопа, фильтр
y += alpha*(x - y) на каждом пакете. Угол и скорость инвертируются вместе (ось IMU с обратным знаком).

Упрощение: одномерная модель. Угол Twist = поворот платформы вокруг оси вала относительно эталона. Из кватерниона
он получается как twist = 2*atan2(n . v, w) для q_rel = conj(q_ref) * q_now (n - ось вала в осях чипа); здесь
этот шаг не моделируется, считается, что ось выбрана верно и swing равен нулю.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class IMUModel:
    def __init__(self, p, h: float):
        self.T = 1.0 / p.f_dmp_hz
        self.delay_steps = int(round(p.meas_delay_ms * 1e-3 / h))
        self.sign = -1.0 if p.invert_meas else 1.0
        self.angle_noise = p.angle_noise_deg
        self.gyro_noise = p.gyro_noise_dps
        self.gyro_bias = p.gyro_bias_dps
        self.a_ang = p.angle_filter_alpha
        self.a_gyr = p.gyro_filter_alpha
        self.rng = np.random.default_rng(p.noise_seed)
        self.buf: deque = deque(maxlen=self.delay_steps + 1)
        self.next_t = 0.0
        self.angle = 0.0
        self.rate = 0.0
        self.packets = 0

    def reset(self, theta0: float, omega0: float = 0.0):
        self.buf = deque([(theta0, omega0)] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        self.next_t = 0.0
        self.angle = self.sign * theta0
        self.rate = self.sign * omega0
        self.packets = 0

    def update(self, t: float, theta_true: float, omega_true: float):
        """Принимает истинные значения текущего подшага; возвращает (угол, скорость, новый_пакет)."""
        self.buf.append((theta_true, omega_true))
        new = False
        while t >= self.next_t - 1e-12:
            th, om = self.buf[0]                       # значение, отстоящее на delay_steps подшагов
            if self.angle_noise > 0.0:
                th = th + self.rng.normal(0.0, self.angle_noise)
            om = om + self.gyro_bias
            if self.gyro_noise > 0.0:
                om = om + self.rng.normal(0.0, self.gyro_noise)
            th *= self.sign
            om *= self.sign
            if self.packets == 0:
                self.angle, self.rate = th, om         # первый пакет: фильтры стартуют от измерения
            else:
                self.angle += self.a_ang * (th - self.angle)
                self.rate += self.a_gyr * (om - self.rate)
            self.packets += 1
            self.next_t += self.T
            new = True
        return self.angle, self.rate, new
