"""Автоматические проверки математической модели и регулятора (без интерфейса).

Запуск:  python -m unittest discover -s tests -v      (pytest не требуется)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gimbal_sim import controllers, export, identification, metrics, params as P, scenarios, servo, servo_tests  # noqa: E402
from gimbal_sim import simulate as S  # noqa: E402


def run(**kw):
    d = P.defaults()
    d.update(kw)
    return S.simulate(d)


class TestParams(unittest.TestCase):
    def test_defaults_valid_and_json_roundtrip(self):
        d = P.defaults()
        p, w = P.validate(d)
        self.assertEqual(w, [])
        self.assertEqual(P.from_json(P.to_json(d)), d)

    def test_invalid_values_rejected(self):
        with self.assertRaises(P.ParamError):
            P.validate({**P.defaults(), "kp_us_per_deg": float("nan")})
        with self.assertRaises(P.ParamError):
            P.validate({**P.defaults(), "kp_us_per_deg": "abc"})
        with self.assertRaises(P.ParamError):
            P.validate({**P.defaults(), "ctrl_mode": "XYZ"})
        with self.assertRaises(P.ParamError):
            P.validate({**P.defaults(), "duration_s": float("inf")})
        with self.assertRaises(P.ParamError):
            P.from_json("{not json")

    def test_out_of_range_is_clamped_with_warning(self):
        p, w = P.validate({**P.defaults(), "kp_us_per_deg": 1e6})
        self.assertEqual(p.kp_us_per_deg, 300.0)
        self.assertTrue(any("KP_US_PER_DEG" in x for x in w))


class TestServoModel(unittest.TestCase):
    def test_reproduces_documented_60_degree_time(self):
        for sup, target in (("4.8", 0.10), ("6.0", 0.08)):
            p, _ = P.validate({**P.defaults(), "supply": sup, "en_frame": False, "en_delay": False, "en_deadband": False})
            self.assertAlmostEqual(servo.time_to_reach(p, 60.0, 1.0), target, delta=0.004)

    def test_servo_suite_has_no_failed_checks(self):
        p, _ = P.validate(P.defaults())
        for t in servo_tests.run_suite(p):
            for txt, ok in t["checks"]:
                self.assertTrue(ok, f"{t['name']}: {txt}")

    def test_no_teleport_and_limits_in_every_model(self):
        for model in ("electromech", "first_order", "ideal"):
            p, _ = P.validate({**P.defaults(), "servo_model": model})
            cal = servo.Calibration(p)
            r = servo.servo_response(p, lambda t: cal.angle_to_pulse(120.0) if t > 0.05 else cal.p_center, 0.6, 5e-4)
            th = np.array(r["angle"])
            self.assertLessEqual(th.max(), cal.hi + 1e-6, model)
            if model != "ideal":
                vmax = servo_tests._speed_bound(p)
                self.assertLessEqual(np.max(np.abs(np.diff(th))) / 5e-4, vmax * 1.02, model)

    def test_command_beyond_range_never_gives_impossible_angle(self):
        p, _ = P.validate({**P.defaults(), "mech_max_deg": 50.0})
        cal = servo.Calibration(p)
        r = servo.servo_response(p, lambda t: 2500.0 if t > 0.05 else 1500.0, 0.6, 5e-4)
        self.assertLessEqual(max(r["angle"]), cal.hi + 1e-6)
        self.assertTrue(any(r["at_stop"]))
        r0 = servo.servo_response(p, lambda t: 2500.0, 0.1, 5e-4)          # старт с командой за упором: вал не стартует за упором
        self.assertLessEqual(max(r0["angle"]), cal.hi + 1e-6)

    def test_disabled_servo_does_not_move(self):
        p, _ = P.validate({**P.defaults(), "servo_enabled": False})
        r = servo.servo_response(p, lambda t: 2000.0 if t > 0.1 else 1500.0, 0.5, 5e-4)
        self.assertTrue(np.allclose(r["angle"], r["angle"][0]))

    def test_in_loop_backlash_delays_reversal_but_output_reaches_target(self):
        base = {**P.defaults(), "en_deadband": False}
        cal = servo.Calibration(P.validate(base)[0])
        prof = lambda t: cal.angle_to_pulse(10.0) if t < 0.05 else cal.angle_to_pulse(8.0)
        times = []
        for lash in (False, True):
            p, _ = P.validate({**base, "en_backlash": lash, "backlash_deg": 3.0})
            r = servo.servo_response(p, prof, 0.6, 5e-4)
            th = np.array(r["angle"])
            self.assertAlmostEqual(th[-1], 8.0, delta=0.1)                      # потенциометр на выходном валу: цель достигнута
            times.append(r["t"][int(np.argmax((np.array(r["t"]) > 0.05) & (th <= 8.3)))])
        self.assertGreater(times[1], times[0])

    def test_load_side_backlash_hides_small_reversal_from_platform(self):
        r = run(en_backlash=True, backlash_load_deg=4.0, backlash_deg=0.0, initial_error_deg=0.0, scenario="initial",
                kp_us_per_deg=0.0, duration_s=1.0)
        self.assertTrue(np.all(r.s["load_angle"] == 0.0))
        lash = servo.LoadLash(P.validate({**P.defaults(), "en_backlash": True, "backlash_load_deg": 4.0})[0])
        lash.reset(0.0)
        seq = [lash.apply(x) for x in (0, 1, 3, 5, 4, 3.5, 3.1, 2.0, 0.5)]
        self.assertEqual(seq[3], 3.0)                                         # вперёд: платформа отстаёт на половину люфта
        self.assertEqual(seq[4], 3.0)                                         # реверс на 1° внутри люфта: платформа стоит
        self.assertEqual(seq[5], 3.0)
        self.assertEqual(seq[7], 3.0)                                         # реверс 5 -> 2 всё ещё внутри люфта 4°
        self.assertAlmostEqual(seq[8], 2.5, places=9)                         # реверс 5 -> 0.5 вышел за люфт: платформа поехала

    def test_load_slows_servo(self):
        p0, _ = P.validate(P.defaults())
        p1, _ = P.validate({**P.defaults(), "j_load_kgm2": 1e-3})
        self.assertGreater(servo.time_to_reach(p1), servo.time_to_reach(p0))

    def test_fractional_delay_is_continuous(self):
        t_first = []
        for dly in (10.0, 10.5, 11.0):
            p, _ = P.validate({**P.defaults(), "en_frame": False, "en_deadband": False, "delay_ms": dly})
            r = servo.servo_response(p, lambda t: 1700.0 if t >= 0.05 else 1500.0, 0.2, 1e-3)
            th = np.array(r["angle"])
            t_first.append(r["t"][int(np.argmax(th > 0.05))])
        self.assertLessEqual(t_first[0], t_first[1])
        self.assertLessEqual(t_first[1], t_first[2])


class TestController(unittest.TestCase):
    def test_p_law_matches_sketch_formula(self):
        p, _ = P.validate(P.defaults())
        c = controllers.Controller(p)
        c.reset(1500)
        pulse, sat = c.update(10.0, 0.0, 0.005)
        self.assertEqual(pulse, controllers.lroundf(1500 - 10.0 * 11.11))
        self.assertFalse(sat)

    def test_zero_kp_gives_no_correction(self):
        p, _ = P.validate({**P.defaults(), "kp_us_per_deg": 0.0})
        c = controllers.Controller(p)
        c.reset(1500)
        for a in (-50.0, 0.0, 17.0):
            self.assertEqual(c.update(a, 3.0, 0.005)[0], 1500)

    def test_target_always_within_limits(self):
        p, _ = P.validate({**P.defaults(), "kp_us_per_deg": 300.0})
        c = controllers.Controller(p)
        c.reset(1500)
        for a in np.linspace(-170, 170, 200):
            pulse, _ = c.update(float(a), 0.0, 0.005)
            self.assertGreaterEqual(pulse, 600)
            self.assertLessEqual(pulse, 2400)
            self.assertLessEqual(abs(pulse - 1500), 900)

    def test_limiter_never_exceeds_step(self):
        p, _ = P.validate(P.defaults())
        lim = controllers.PulseLimiter(p)
        lim.reset(1500)
        prev = 1500
        for i in range(2000):
            lim.set_target(2400 if (i // 300) % 2 == 0 else 600)
            cur, _ = lim.step(i * 0.001)
            self.assertLessEqual(abs(cur - prev), 10)
            prev = cur

    def test_cascade_antiwindup_keeps_state_in_range(self):
        p, _ = P.validate({**P.defaults(), "ctrl_mode": "cascade", "kw_us_per_dps": 200.0})
        c = controllers.Controller(p)
        c.reset(1500)
        for _ in range(500):
            c.update(60.0, 0.0, 0.05)
        self.assertGreaterEqual(c.corr_state, c.corr_lo - 1e-9)
        self.assertLessEqual(c.corr_state, c.corr_hi + 1e-9)


class TestClosedLoop(unittest.TestCase):
    def test_zero_error_stays_at_zero(self):
        r = run(initial_error_deg=0.0, duration_s=5.0)
        self.assertLess(np.max(np.abs(r.s["theta_true"])), 1e-9)
        self.assertTrue(np.all(r.s["actual_pulse"] == 1500))

    def test_initial_error_is_state_not_disturbance(self):
        r = run(initial_error_deg=30.0, duration_s=6.0)
        self.assertAlmostEqual(r.s["theta_true"][0], 30.0, places=6)
        self.assertTrue(np.all(r.s["base_angle"] == 0.0))             # основание не двигалось
        self.assertLess(abs(r.s["theta_true"][-1]), 1.0)              # вернулась к нулю

    def test_disabled_servo_angle_unchanged_by_commands(self):
        r = run(servo_enabled=False, initial_error_deg=20.0, duration_s=4.0)
        self.assertTrue(np.allclose(r.s["servo_angle"], r.s["servo_angle"][0]))
        self.assertTrue(np.allclose(r.s["theta_true"], r.s["theta_true"][0]))
        self.assertTrue(np.any(r.s["actual_pulse"] != 1500))          # команды при этом выдавались

    def test_target_pulse_within_allowed_range(self):
        for kp in (5.0, 18.0, 300.0):
            r = run(kp_us_per_deg=kp, initial_error_deg=60.0, duration_s=5.0)
            self.assertGreaterEqual(r.s["target_pulse"].min(), 600)
            self.assertLessEqual(r.s["target_pulse"].max(), 2400)
            self.assertLessEqual(np.max(np.abs(r.s["target_pulse"] - 1500)), 900)

    def test_command_rate_limit_respected(self):
        r = run(initial_error_deg=60.0, duration_s=5.0)
        a = r.s["actual_pulse"]
        w = 4                                                         # 4 * 5 мс = 20 мс
        self.assertLessEqual(np.max(np.abs(a[w:] - a[:-w])), 10.0 + 1e-9)

    def test_zero_kp_gives_no_proportional_correction(self):
        r = run(kp_us_per_deg=0.0, initial_error_deg=30.0, duration_s=3.0)
        self.assertTrue(np.all(r.s["target_pulse"][2:] == 1500))       # с первого обновления команды: ровно центр
        self.assertAlmostEqual(r.s["theta_true"][-1], r.s["servo_angle"][-1], places=6)

    def test_sign_symmetry(self):
        a = run(initial_error_deg=20.0, duration_s=4.0)
        b = run(initial_error_deg=-20.0, duration_s=4.0)
        self.assertTrue(np.allclose(a.s["target_pulse"] - 1500, -(b.s["target_pulse"] - 1500), atol=1.0))
        self.assertTrue(np.allclose(a.s["theta_true"], -b.s["theta_true"], atol=1e-6))

    def test_determinism_without_noise_and_with_seeded_noise(self):
        a = run(duration_s=3.0)
        b = run(duration_s=3.0)
        self.assertTrue(np.array_equal(a.s["theta_true"], b.s["theta_true"]))
        c = run(duration_s=3.0, angle_noise_deg=0.1, gyro_noise_dps=0.5, noise_seed=7)
        d = run(duration_s=3.0, angle_noise_deg=0.1, gyro_noise_dps=0.5, noise_seed=7)
        e = run(duration_s=3.0, angle_noise_deg=0.1, gyro_noise_dps=0.5, noise_seed=8)
        self.assertTrue(np.array_equal(c.s["theta_true"], d.s["theta_true"]))
        self.assertFalse(np.array_equal(c.s["theta_meas"], e.s["theta_meas"]))

    def test_positive_feedback_is_not_masked(self):
        r = run(invert_cmd=True, kp_us_per_deg=18.0, initial_error_deg=10.0, duration_s=10.0)
        self.assertLess(r.info["K_eff_P"], 0)
        m = {x["key"]: x for x in metrics.compute_metrics(r)}
        self.assertIn("pos_fb", m)
        self.assertGreater(np.max(np.abs(r.s["theta_true"])), 40.0)   # уходит к упору, а не «стабилизируется»
        self.assertGreater(m["sat_cmd"]["value"] + m["sat_stop"]["value"], 10.0)
        self.assertIn(m["settle"]["value"], (None,))                   # не установилась

    def test_inverted_measurement_also_flips_feedback(self):
        r = run(invert_meas=True, kp_us_per_deg=18.0, initial_error_deg=10.0, duration_s=6.0)
        self.assertLess(r.info["K_eff_P"], 0)
        r2 = run(invert_meas=True, invert_cmd=True, kp_us_per_deg=18.0, initial_error_deg=10.0, duration_s=6.0)
        self.assertGreater(r2.info["K_eff_P"], 0)                      # две инверсии компенсируют друг друга
        self.assertLess(abs(r2.s["theta_true"][-1]), 1.0)

    def test_static_residual_matches_1_over_1_plus_K(self):
        K = 1.5
        kp = K * 11.1111111
        r = run(initial_error_deg=0.0, scenario="step", dist_amp_deg=10.0, dist_t0_s=1.0, kp_us_per_deg=kp,
                rate_limit_enable=False, en_deadband=False, duration_s=10.0)
        self.assertAlmostEqual(r.s["theta_true"][-1], 10.0 / (1 + K), delta=0.1)

    def test_slow_update_flipflop_threshold_is_K_equal_one(self):
        kw = dict(control_period_preset="5000", duration_s=120.0, rate_limit_enable=True, initial_error_deg=10.0)
        low = metrics.classify_oscillation(*self._te(run(kp_us_per_deg=10.0, **kw)))       # K=0.9: медленно затухает
        high = metrics.classify_oscillation(*self._te(run(kp_us_per_deg=15.0, **kw)))     # K=1.35: растёт до насыщения
        gone = metrics.classify_oscillation(*self._te(run(kp_us_per_deg=6.0, **kw)))      # K=0.54: за 60 с затухло полностью
        self.assertEqual(gone["label"], "колебаний нет")
        self.assertIn("затухающие", low["label"])
        self.assertTrue("нарастающие" in high["label"] or "насыщением" in high["label"], high["label"])

    @staticmethod
    def _te(res):
        return res.t, res.s["theta_true"], res.s["cmd_sat"] + res.s["at_stop"], res.params.band_deg

    def test_unreachable_initial_error_is_reported(self):
        r = run(initial_error_deg=120.0, duration_s=2.0)
        self.assertTrue(any("недостижима" in w for w in r.warnings))
        self.assertLessEqual(r.s["servo_angle"][0], 90.0 + 1e-6)

    def test_servo_speed_in_loop_not_above_motor_limit(self):
        r = run(initial_error_deg=60.0, rate_limit_enable=False, duration_s=3.0, kp_us_per_deg=40.0)
        self.assertLessEqual(np.max(np.abs(r.s["servo_omega"])), servo.resolve_omega_nl(r.params) * 1.05)

    def test_all_outputs_finite_in_extreme_valid_setup(self):
        r = run(kp_us_per_deg=300.0, scenario="random", dist_amp_deg=30.0, angle_noise_deg=2.0, gyro_noise_dps=10.0,
                en_backlash=True, backlash_deg=5.0, en_friction=True, coulomb_nm=0.05, duration_s=5.0)
        for k, v in r.s.items():
            self.assertTrue(np.all(np.isfinite(v)), k)

    def test_scenarios_produce_expected_base_motion(self):
        p, _ = P.validate({**P.defaults(), "scenario": "step", "dist_amp_deg": 12.0, "dist_t0_s": 1.0})
        th, om, al = scenarios.base_signal(p, 4000, 0.001)
        self.assertAlmostEqual(th[-1], 12.0, places=6)
        self.assertAlmostEqual(th[0], 0.0, places=9)
        p, _ = P.validate({**P.defaults(), "scenario": "pulse", "dist_amp_deg": 8.0, "dist_t0_s": 1.0, "dist_dur_s": 0.5})
        th, _, _ = scenarios.base_signal(p, 4000, 0.001)
        self.assertAlmostEqual(th[-1], 0.0, places=6)
        self.assertAlmostEqual(th.max(), 8.0, places=3)
        p, _ = P.validate({**P.defaults(), "scenario": "random", "dist_amp_deg": 5.0})
        a, _, _ = scenarios.base_signal(p, 20000, 0.001)
        b, _, _ = scenarios.base_signal(p, 20000, 0.001)
        self.assertTrue(np.array_equal(a, b))
        self.assertAlmostEqual(float(np.std(a)), 5.0, delta=0.05)

    def test_compare_uses_identical_conditions(self):
        res = S.simulate_kp_list({**P.defaults(), "duration_s": 3.0, "rate_limit_enable": False}, [5, 11.11, 18])
        self.assertEqual(len(res), 3)
        for r in res:
            self.assertAlmostEqual(r.s["theta_true"][0], 30.0, places=9)
        self.assertNotEqual(res[0].s["theta_true"][100], res[2].s["theta_true"][100])


class TestExportAndIdentification(unittest.TestCase):
    def test_csv_has_required_columns_and_finite_values(self):
        r = run(duration_s=2.0)
        df = export.result_to_dataframe(r)
        for col in ("time_s", "theta_true_deg", "theta_meas_deg", "omega_meas_dps", "target_pulse_us",
                    "actual_pulse_us", "servo_angle_deg", "control_error_deg", "disturbance_deg"):
            self.assertIn(col, df.columns)
        self.assertTrue(np.isfinite(df.to_numpy(dtype=float)).all())
        self.assertGreater(len(export.csv_bytes(r)), 1000)

    def test_identification_recovers_known_servo(self):
        true = {**P.defaults(), "servo_model": "electromech", "auto_omega_nl": False, "en_frame": False, "en_delay": True,
                "en_deadband": False, "tau_a_ms": 9.0, "e_sat_deg": 15.0, "omega_nl_dps": 820.0, "delay_ms": 6.0}
        pt, _ = P.validate(true)
        cal = servo.Calibration(pt)
        rng = np.random.default_rng(1)
        ds = []
        for amp in (60.0, 8.0):
            p1, p0 = cal.angle_to_pulse(amp), cal.p_center
            r = servo.servo_response(pt, lambda t, a=p1, b=p0: a if t >= 0.05 else b, 0.6, 1e-3)
            t = np.arange(0, 0.6, 0.004)
            ds.append(identification.StepDataset(t, np.interp(t, r["t"], r["angle"]) + rng.normal(0, 0.15, t.size), p0, p1, 0.05))
        fit = identification.fit_servo(P.defaults(), ds)
        self.assertLess(fit["rmse_after"], 0.25)
        for k in ("tau_a_ms", "e_sat_deg", "omega_nl_dps"):
            self.assertAlmostEqual(fit["params"][k] / true[k], 1.0, delta=0.12, msg=k)

    def test_measurement_csv_parser(self):
        t, a = identification.load_measurement_csv("time_s;angle\n0;0\n0,1;1,5\n0.2;4\n0.3;7\n0.4;9\n0.5;10\n")
        self.assertEqual(len(t), 6)
        self.assertAlmostEqual(a[1], 1.5)
        with self.assertRaises(ValueError):
            identification.load_measurement_csv("a,b\n1,2\n")

    def test_model_step_characteristics(self):
        ch = identification.model_step_characteristics(P.defaults(), 1500.0, 1500.0 + 60 * 11.1111)
        self.assertAlmostEqual(ch["amplitude_deg"], 60.0, delta=0.5)
        self.assertGreater(ch["max_speed_dps"], 300.0)
        self.assertGreater(ch["reach_95_s"], ch["start_delay_s"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
