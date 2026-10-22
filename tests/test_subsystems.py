"""Subsystem-level tests on the integrated spacecraft."""
import unittest

import numpy as np

from edgesat.ai.cnn import Int8CNN, macs
from edgesat.ai.scenes import sample_training_set
from edgesat.core.constants import DEG
from edgesat.sim import Simulation, load_config
from edgesat.spacecraft import ROOT
from edgesat.subsystems.comms import Comms
from edgesat.subsystems.payload import DataStore, Product


class TestNominalOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulation()
        cls.sim.run(3 * 3600)

    def test_stays_nominal_and_healthy(self):
        sc = self.sim.sc
        self.assertEqual(sc.obdh.mode, "NOMINAL")
        self.assertGreater(sc.eps.soc, 0.5)
        for name, T in sc.tcs.temps_c().items():
            self.assertTrue(-100 < T < 120, name)

    def test_estimator_accuracy(self):
        # star tracker 12" 1-sigma: knowledge should stay at arcsecond level (or degrade
        # gracefully when the tracker is excluded)
        self.assertLess(self.sim.sc.adcs.tm["knowledge_error_arcsec"], 600)

    def test_mean_altitude(self):
        self.assertAlmostEqual(self.sim.sc.obdh.sma_mean / 1000 - 6378.137, 525.0, delta=1.0)

    def test_no_nan_in_snapshot(self):
        import json
        json.dumps(self.sim.snapshot(), allow_nan=False, default=float)


class TestFDIR(unittest.TestCase):
    def test_wheel_seizure_is_isolated(self):
        sim = Simulation()
        sim.run(600)
        sim.command("inject", "wheel_x_seize")
        sim.run(120)
        sc = sim.sc
        self.assertTrue(sc.adcs.faults.wheel_isolated[0])
        self.assertEqual(sc.obdh.mode, "SAFE")
        sim.run(1500)
        self.assertEqual(sc.obdh.mode, "NOMINAL")       # autonomous recovery in degraded mode
        self.assertLess(sc.adcs.tm["pointing_error_deg"], 2.0)

    def test_thruster_leak_latched(self):
        sim = Simulation()
        sim.run(60)
        sim.command("inject", "thruster_leak")
        sim.run(120)
        self.assertTrue(sim.sc.prop.latch_closed)

    def test_seu_reboot_and_recovery(self):
        sim = Simulation()
        sim.run(60)
        sim.command("inject", "obc_seu")
        sim.run(30)
        self.assertEqual(sim.sc.obdh.mode, "REBOOT")
        sim.run(1200)
        self.assertEqual(sim.sc.obdh.mode, "NOMINAL")

    def test_star_tracker_failure_degrades_knowledge(self):
        sim = Simulation()
        sim.run(300)
        sig_before = max(sim.sc.adcs.tm["sigma3_arcsec"])
        sim.command("inject", "star_tracker_fail")
        sim.run(1200)
        self.assertGreater(max(sim.sc.adcs.tm["sigma3_arcsec"]), 3 * sig_before)


class TestLEOP(unittest.TestCase):
    def test_detumble_and_acquire(self):
        sim = Simulation(scenario="leop")
        sim.run(2.5 * 3600)
        self.assertEqual(sim.sc.obdh.mode, "NOMINAL")
        self.assertLess(sim.sc.adcs.tm["rate_norm_deg_s"], 0.5)


class TestComms(unittest.TestCase):
    def test_free_space_loss_and_acm(self):
        c = Comms(load_config())
        hi = c.budget(80 * DEG, 530e3, 0.0, 10.5)
        lo = c.budget(8 * DEG, 2200e3, 60 * DEG, 10.5)
        self.assertAlmostEqual(hi["fspl_db"], 20 * np.log10(4 * np.pi * 530e3 * 2.245e9 / 299792458), places=6)
        self.assertGreater(hi["rate_bps"], lo["rate_bps"])
        self.assertGreaterEqual(hi["margin_db"], 3.0)


class TestDataStore(unittest.TestCase):
    def test_priority_and_eviction(self):
        st = DataStore(100.0, "ai")
        st.add(Product(1, "FRAME", 40, 3, 0.2, 0.0))
        st.add(Product(2, "FRAME", 40, 3, 0.9, 1.0))
        st.add(Product(3, "ALERT", 1, 1, 5.0, 2.0, fire_ids=(7,)))
        self.assertTrue(st.add(Product(4, "FRAME", 40, 3, 0.95, 3.0)))   # evicts the 0.2 frame
        self.assertNotIn(1, st.items)
        st.transmit(1, 10.0, {})
        self.assertEqual(st.stats.alerts_delivered, 1)                  # alert goes first
        self.assertEqual(st.stats.fire_alert_ground_t[7], 10.0)

    def test_fifo_drops_when_full(self):
        st = DataStore(50.0, "fifo")
        self.assertTrue(st.add(Product(1, "FRAME", 40, 3, 0.0, 0.0)))
        self.assertFalse(st.add(Product(2, "FRAME", 40, 3, 0.0, 1.0)))
        self.assertEqual(st.stats.dropped_memory_full, 1)


class TestEdgeAI(unittest.TestCase):
    def test_int8_model(self):
        m = Int8CNN.load(ROOT / "edgesat" / "ai" / "model_int8.npz")
        X, y = sample_training_set(600, seed=1234)
        acc = (m.forward(X).argmax(1) == y).mean()
        self.assertGreater(acc, 0.9)
        self.assertLess(m.size_bytes, 8 * 1024)
        self.assertEqual(macs(), 737408)


if __name__ == "__main__":
    unittest.main()
