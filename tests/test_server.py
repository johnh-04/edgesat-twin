"""Server API tests (no browser)."""
import json
import time
import unittest

from starlette.testclient import TestClient

import server.app as app_mod


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app_mod.create_app(warp=200)
        cls.client = TestClient(cls.app)
        time.sleep(2.0)

    def test_index_and_static(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/js/app.js").status_code, 200)
        self.assertEqual(self.client.get("/api/land").status_code, 200)

    def test_info(self):
        info = self.client.get("/api/info").json()
        self.assertIn("faults", info)
        self.assertEqual(info["stations"][0]["name"], "Bari (Poliba)")

    def test_snapshot_and_commands(self):
        snap = json.loads(app_mod.runner.snapshot)
        self.assertGreater(snap["t"], 0)
        r = self.client.post("/api/command", json={"cmd": "inject", "arg": "gyro_bias_jump"})
        self.assertTrue(r.json()["ok"])
        r = self.client.post("/api/command", json={"cmd": "nope"})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/command", json={"cmd": "warp", "arg": 10})
        self.assertEqual(app_mod.runner.warp, 10)


if __name__ == "__main__":
    unittest.main()
