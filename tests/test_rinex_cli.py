"""RINEX 采集命令行入口测试。"""
import json
import unittest
from pathlib import Path

from capture_data.rinex_cli import main

DATA = Path(__file__).resolve().parent.parent / "data"


class RinexCliTests(unittest.TestCase):
    def test_manifest_entry_emits_versioned_epochs(self):
        payload = _capture(main, ["rinex-manifest", str(DATA / "normal.json")])
        self.assertEqual(len(payload["files"]), 1)
        result = payload["files"][0]
        self.assertEqual(result["header"]["version"], 3.04)
        flags = [(e["flag"], e["catalog_version"]) for e in result["epochs"]]
        self.assertEqual(flags, [(0, 0), (4, 1), (0, 1), (6, 1)])
        # 每条观测都带系统、观测码、字节范围与头版本。
        first_obs = result["epochs"][0]["satellites"][0]["observations"][0]
        for key in ("system", "code", "byte_start", "byte_end", "catalog_version"):
            self.assertIn(key, first_obs)

    def test_file_entry_with_borrowed_leap_table(self):
        payload = _capture(main, [
            "rinex-file", str(DATA / "utc-leap.rnx"),
            "--leap", str(DATA / "utc-positive-leap.json"),
        ])
        epochs = payload["epochs"]
        self.assertEqual(epochs[0]["time"]["second_text"].strip(), "60.0000000")
        self.assertEqual(epochs[0]["time"]["time_axis_status"], "ok")
        # 闰秒两历元在 TAI 轴上相差 1 秒。
        self.assertEqual(
            epochs[0]["time"]["tai_nanoseconds"] + 1_000_000_000,
            epochs[1]["time"]["tai_nanoseconds"])

    def test_manifest_identity_failure_propagates(self):
        with self.assertRaises(ValueError):
            main(["rinex-manifest", str(DATA / "invalid-identity.json")])


def _capture(func, argv):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        func(argv)
    return json.loads(buf.getvalue())


if __name__ == "__main__":
    unittest.main()
