"""Unit tests of the mathematical core: attitude, orbit, environment."""
import unittest
from datetime import datetime

import numpy as np

from edgesat.core.attitude import (attitude_error, dcm_from_quat, propagate_quat, qmul, qnormalize,
                                   quat_from_axis_angle, quat_from_dcm, skew, triad)
from edgesat.core.constants import DEG, MU_EARTH, R_EARTH
from edgesat.core.environment import (atmospheric_density, illumination_fraction, magnetic_field_ecef,
                                      sun_position_eci, LandMask)
from edgesat.core.orbit import (OrbitPropagator, coe_to_rv, ecef_to_geodetic, geodetic_to_ecef, julian_date,
                                look_angles, rv_to_coe, sso_inclination)


class TestAttitude(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.q = qnormalize(rng.normal(size=4))
        self.p = qnormalize(rng.normal(size=4))

    def test_composition_matches_dcm_product(self):
        np.testing.assert_allclose(dcm_from_quat(qmul(self.q, self.p)),
                                   dcm_from_quat(self.q) @ dcm_from_quat(self.p), atol=1e-12)

    def test_dcm_roundtrip(self):
        np.testing.assert_allclose(quat_from_dcm(dcm_from_quat(self.q)), self.q, atol=1e-12)

    def test_kinematics_dcm_derivative(self):
        w = np.array([0.02, -0.01, 0.03])
        dt = 1e-4
        A0 = dcm_from_quat(self.q)
        A1 = dcm_from_quat(propagate_quat(self.q, w, dt))
        np.testing.assert_allclose((A1 - A0) / dt, -skew(w) @ A0, atol=1e-5)

    def test_frame_rotation_sign(self):
        A = dcm_from_quat(quat_from_axis_angle(np.array([0, 0, 1.0]), np.pi / 2))
        np.testing.assert_allclose(A @ [1, 0, 0], [0, -1, 0], atol=1e-12)

    def test_small_error_vector(self):
        dq = quat_from_axis_angle(np.array([1.0, 0, 0]), 0.01)
        err = attitude_error(qmul(dq, self.q), self.q)
        np.testing.assert_allclose(err, [0.01, 0, 0], atol=1e-6)

    def test_triad(self):
        A = triad(np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0.3, 1.0]))
        np.testing.assert_allclose(A @ [1, 0, 0], [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(A), 1.0)


class TestOrbit(unittest.TestCase):
    def test_sso_inclination(self):
        self.assertAlmostEqual(np.degrees(sso_inclination(R_EARTH + 525e3)), 97.50, places=1)

    def test_elements_roundtrip(self):
        r, v = coe_to_rv(7000e3, 0.01, 1.0, 0.5, 0.3, 0.2)
        c = rv_to_coe(r, v)
        self.assertAlmostEqual(c["a"], 7000e3, delta=1e-3)
        self.assertAlmostEqual(c["e"], 0.01, places=9)
        self.assertAlmostEqual(c["i"], 1.0, places=9)

    def test_energy_conservation_two_body_plus_j2(self):
        r, v = coe_to_rv(R_EARTH + 525e3, 0.001, 97.5 * DEG, 0.0, 0.0, 0.0)
        prop = OrbitPropagator(r, v)

        def energy(r, v):
            from edgesat.core.constants import J2
            rn = np.linalg.norm(r)
            u_j2 = MU_EARTH * J2 * R_EARTH ** 2 / (2 * rn ** 3) * (3 * (r[2] / rn) ** 2 - 1)
            return 0.5 * v @ v - MU_EARTH / rn + u_j2
        e0 = energy(prop.r, prop.v)
        for _ in range(5700):
            prop.step(1.0, np.zeros(3))
        self.assertLess(abs(energy(prop.r, prop.v) - e0) / abs(e0), 1e-9)

    def test_geodetic_roundtrip(self):
        r = geodetic_to_ecef(41.1 * DEG, 16.9 * DEG, 30.0)
        lat, lon, alt = ecef_to_geodetic(r)
        self.assertAlmostEqual(lat, 41.1 * DEG, places=9)
        self.assertAlmostEqual(lon, 16.9 * DEG, places=9)
        self.assertAlmostEqual(alt, 30.0, places=3)

    def test_zenith_pass(self):
        gs = geodetic_to_ecef(0.7, 0.3, 0.0)
        sat = geodetic_to_ecef(0.7, 0.3, 500e3)
        el, _, rng = look_angles(sat, gs, 0.7, 0.3)
        self.assertAlmostEqual(el, np.pi / 2, places=6)
        self.assertAlmostEqual(rng, 500e3, delta=1.0)


class TestEnvironment(unittest.TestCase):
    def test_sun_declination_solstice(self):
        jd = julian_date(datetime(2027, 6, 21, 12))
        s = sun_position_eci(jd)
        self.assertAlmostEqual(np.degrees(np.arcsin(s[2] / np.linalg.norm(s))), 23.44, delta=0.05)

    def test_eclipse(self):
        sun = np.array([1.496e11, 0, 0])
        self.assertEqual(illumination_fraction(np.array([7e6, 0, 0]), sun), 1.0)
        self.assertEqual(illumination_fraction(np.array([-7e6, 0, 0]), sun), 0.0)

    def test_magnetic_field_magnitude(self):
        b_eq = np.linalg.norm(magnetic_field_ecef(np.array([R_EARTH, 0, 0])))
        b_pole = np.linalg.norm(magnetic_field_ecef(np.array([0, 0, R_EARTH])))
        self.assertTrue(25e-6 < b_eq < 35e-6)
        self.assertTrue(55e-6 < b_pole < 65e-6)

    def test_density(self):
        self.assertTrue(1e-13 < atmospheric_density(525e3) < 1e-12)
        self.assertGreater(atmospheric_density(400e3), atmospheric_density(500e3))

    def test_land_mask(self):
        lm = LandMask()
        self.assertTrue(lm.is_land(41.1, 16.9))      # Bari
        self.assertFalse(lm.is_land(38.0, 18.0))     # Ionian Sea
        self.assertFalse(lm.is_land(0.0, -140.0))    # Pacific


if __name__ == "__main__":
    unittest.main()
