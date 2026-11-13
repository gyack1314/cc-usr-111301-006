"""RINEX 3.04 观测采集：固定列、动态目录版本、事件与闰秒。"""
import json
import unittest
from pathlib import Path

from capture_data.rinex3 import (
    LeapTable,
    LeapSecondUnknown,
    ObservationCollector,
    RinexError,
    _seconds_to_integer_nanoseconds,
    civil_nanoseconds,
    gps_to_tai_ns,
    parse_observation_number,
    parse_bytes,
    parse_file,
)

DATA = Path(__file__).resolve().parent.parent / "data"
NS = 1_000_000_000


def obs_line(satellite: str, slots: list[str]) -> str:
    """按 3 列卫星 ID + 每槽 16 字符（14 数值 + LLI + SSI）拼观测行。"""
    body = "".join(s.ljust(16) for s in slots)
    return satellite + body


def head_line(value: str, label: str) -> str:
    return value.ljust(60) + label.ljust(20)


def load_manifest(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def leap(manifest):
    return LeapTable(manifest["leap_table"], manifest["leap_valid_until"])


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("normal.json")
        self.header, self.epochs = parse_file(
            DATA / "observations.rnx", leap(self.m), self.m["max_epoch_satellites"])

    def test_fixed_column_header(self):
        self.assertEqual(self.header.version, 3.04)
        self.assertEqual(self.header.file_type, "O")
        self.assertEqual(self.header.satellite_system, "M")
        self.assertEqual(self.header.clock_offset_applied, 0)
        tofo = self.header.time_of_first_obs
        self.assertEqual((tofo["year"], tofo["month"], tofo["day"]), (2026, 10, 27))
        self.assertEqual(tofo["time_system"], "GPS")
        self.assertEqual(self.header.catalogs, {"G": ["C1C", "L1C", "S1C"]})

    def test_rejects_non_observation_and_bad_version(self):
        good = (DATA / "observations.rnx").read_bytes()
        # 文件类型列（0 基 20）改成 N（导航电文）。
        nav = good[:20] + b"N" + good[21:]
        with self.assertRaises(RinexError):
            parse_bytes(nav)
        # 版本改成 2.11。
        old = b"2.11".rjust(9) + good[9:]
        with self.assertRaises(RinexError):
            parse_bytes(old)


class CatalogVersionTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("normal.json")
        self.header, self.epochs = parse_file(
            DATA / "observations.rnx", leap(self.m), self.m["max_epoch_satellites"])

    def test_flag4_atomic_update_versions_epochs(self):
        flags = [e.flag for e in self.epochs]
        self.assertEqual(flags, [0, 4, 0, 6])
        first, update, after, slip = self.epochs
        # 更新前目录版本 0，只有 3 个观测码。
        self.assertEqual(first.catalog_version, 0)
        self.assertEqual(len(first.satellites[0].slots), 3)
        # flag 4 自身不产生卫星观测，只携带头更新与新版本号。
        self.assertEqual(update.catalog_version, 1)
        self.assertIsNone(update.satellites or None)
        self.assertEqual(update.header_update["from_version"], 0)
        self.assertEqual(update.header_update["to_version"], 1)
        self.assertEqual(update.header_update["catalogs"]["G"],
                         ["C1C", "L1C", "S1C", "D1C"])
        # 更新后同一卫星 G01 沿用新版本，多出 D1C 槽。
        self.assertEqual(after.catalog_version, 1)
        sat = after.satellites[0]
        self.assertEqual(sat.satellite, "G01")
        self.assertEqual([o.code for o in sat.slots],
                         ["C1C", "L1C", "S1C", "D1C"])
        self.assertEqual(sat.physical_slots, 4)

    def test_same_satellite_before_and_after_update(self):
        first, _, after, _ = self.epochs
        before = {o.code: o for o in first.satellites[0].slots}
        after_map = {o.code: o for o in after.satellites[0].slots}
        self.assertEqual(first.satellites[0].satellite,
                         after.satellites[0].satellite)
        # 伪距 C1C 与信噪比 S1C 是不同观测码，绝不能把 SSI/信噪比当伪距。
        self.assertAlmostEqual(before["C1C"].value, 20200000.0)
        self.assertAlmostEqual(before["S1C"].value, 45.0)
        self.assertIsNone(before["L1C"].value)
        self.assertAlmostEqual(after_map["D1C"].value, -2.0)
        # 每条观测都带当时的头版本。
        self.assertTrue(all(o.catalog_version == 0 for o in before.values()))
        self.assertTrue(all(o.catalog_version == 1 for o in after_map.values()))

    def test_failed_flag4_keeps_old_catalog_atomically(self):
        good = (DATA / "observations.rnx").read_bytes()
        # 把 flag4 声明 4 个类型却删掉 D1C，候选目录不完整，必须整体失败，
        # 不能留下“半套”观测类型。
        broken = good.replace(
            b"G    4 C1C L1C S1C D1C",
            b"G    4 C1C L1C S1C    ")
        with self.assertRaises(RinexError):
            parse_bytes(broken, leap(self.m))

    def test_header_continuation_lines(self):
        def head_line(value: str, label: str) -> str:
            return value.ljust(60) + label.ljust(20)
        rnx = "\n".join([
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G   15 C1C L1C S1C D1C C1W L1W S1W C2W L2W S2W C2L L2L S2L",
                      "SYS / # / OBS TYPES "),
            head_line("        C5Q L5Q",
                      "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
        ]) + "\n"
        header, _ = parse_bytes(rnx.encode("utf-8"))
        self.assertEqual(len(header.catalogs["G"]), 15)
        self.assertEqual(header.catalogs["G"][-2:], ["C5Q", "L5Q"])


class SlotAndValueTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("normal.json")
        _, self.epochs = parse_file(
            DATA / "observations.rnx", leap(self.m), self.m["max_epoch_satellites"])

    def test_slots_are_positional_not_split(self):
        first = self.epochs[0]
        obs = first.satellites[0].slots
        codes = [o.code for o in obs]
        self.assertEqual(codes, ["C1C", "L1C", "S1C"])
        # 中间 L1C 槽全空：value/LLI/SSI 都是 None，而不是 0，且没有移位。
        self.assertIsNone(obs[1].value)
        self.assertIsNone(obs[1].lli)
        self.assertIsNone(obs[1].ssi)
        # LLI=1、SSI=7 与数值分开保留在第一槽。
        self.assertEqual(obs[0].lli, 1)
        self.assertEqual(obs[0].ssi, 7)

    def test_original_slot_byte_ranges_round_trip(self):
        raw = (DATA / "observations.rnx").read_bytes()
        first = self.epochs[0]
        for obs in first.observations():
            chunk = raw[obs.byte_start:obs.byte_end]
            self.assertEqual(len(chunk), 16)
            value_chunk = raw[obs.value_byte_start:obs.value_byte_end]
            self.assertEqual(len(value_chunk), 14)
            if obs.value is not None:
                self.assertIn(str(int(obs.value)).encode(), value_chunk
                              if obs.value == int(obs.value) else value_chunk)
        # 精确核验第一槽字节区间对应文件 444:460。
        c1c = first.satellites[0].slots[0]
        self.assertEqual(raw[c1c.byte_start:c1c.byte_end], b"  20200000.00017")

    def test_missing_slot_marks_epoch_incomplete_but_keeps_positions(self):
        m = load_manifest("truncated-observation-slots.json")
        _, epochs = parse_file(DATA / "missing-slot.rnx", leap(m),
                               m["max_epoch_satellites"])
        truncated = epochs[2]
        self.assertFalse(truncated.complete)
        self.assertIn("物理槽 3 与目录 4", truncated.incomplete_reason)
        sat = truncated.satellites[0]
        self.assertEqual(sat.physical_slots, 3)
        # 只保留 3 个槽，但观测码仍按目录前 3 个对齐，缺测位置可还原。
        self.assertEqual([o.code for o in sat.slots], ["C1C", "L1C", "S1C"])
        # 后续 flag 6 历元仍完整，目录版本不回退。
        self.assertTrue(epochs[3].complete)
        self.assertEqual(epochs[3].catalog_version, 1)

    def test_d_exponent_and_strict_number_syntax(self):
        self.assertEqual(parse_observation_number("  1.234D+03    ", 1), 1234.0)
        self.assertEqual(parse_observation_number(" -1.234E-02   ", 1), -0.01234)
        self.assertIsNone(parse_observation_number(" " * 14, 1))
        for bad in ("          nan ", "          inf ", "   12  34     ", "0x10          "):
            with self.assertRaises(RinexError):
                parse_observation_number(bad, 1)

    def test_illegal_lli_ssi_character_rejected(self):
        good = (DATA / "observations.rnx").read_bytes()
        # 第一槽 LLI 列（文件偏移 444+14=458）放入字母。
        bad = good[:458] + b"X" + good[459:]
        with self.assertRaises(RinexError):
            parse_bytes(bad, leap(self.m))


class EventAndCycleSlipTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("normal.json")
        _, self.epochs = parse_file(
            DATA / "observations.rnx", leap(self.m), self.m["max_epoch_satellites"])

    def test_flag6_is_separate_cycle_slip_record(self):
        slip = self.epochs[3]
        self.assertEqual(slip.flag, 6)
        self.assertTrue(slip.cycle_slip)
        self.assertEqual(slip.catalog_version, 1)
        # 周跳记录不进普通观测集合语义：仍按该时刻系统目录定槽。
        self.assertEqual(len(slip.satellites[0].slots), 4)
        normal_flags = [e.flag for e in self.epochs if not e.cycle_slip
                        and e.flag not in (4,)]
        self.assertTrue(all(f in (0, 1) for f in normal_flags))

    def test_event_flags_do_not_count_comments_as_satellites(self):
        rnx = "\n".join([
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G    2 C1C L1C", "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            "> 2026 10 27 00 00  0.0000000  3  2",
            head_line("Receiver starts now", "COMMENT"),
            head_line("Antenna changed", "COMMENT"),
            "> 2026 10 27 00 00  1.0000000  0  1",
            obs_line("G01", ["      10.0000", "      20.0000"]),
        ]) + "\n"
        _, epochs = parse_bytes(rnx.encode("utf-8"), leap(self.m))
        event, obs = epochs
        self.assertEqual(event.flag, 3)
        self.assertEqual(event.declared_satellites, 2)
        self.assertEqual(len(event.event_lines), 2)
        self.assertEqual(event.satellites, ())
        # 注释行绝不能被当成卫星；下一历元正常解析。
        self.assertEqual(obs.flag, 0)
        self.assertEqual(obs.satellites[0].satellite, "G01")

    def test_duplicate_satellite_in_epoch_rejected(self):
        rnx = "\n".join([
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G    1 C1C", "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            "> 2026 10 27 00 00  0.0000000  0  2",
            obs_line("G01", ["      10.0000"]),
            obs_line("G01", ["      11.0000"]),
        ]) + "\n"
        with self.assertRaises(RinexError):
            parse_bytes(rnx.encode("utf-8"), leap(self.m))


class TimeAxisTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("utc-positive-leap.json")
        self.table = leap(self.m)

    def test_explicit_utc_leap_second_two_epochs(self):
        _, epochs = parse_file(DATA / "utc-leap.rnx", self.table)
        leap_epoch, after = epochs
        # 原秒字段保留显式闰秒 60。
        self.assertEqual(leap_epoch.second_text.strip(), "60.0000000")
        self.assertEqual(after.second_text.strip(), "0.0000000")
        # 共同 TAI 整数纳秒轴：闰秒与下一秒相差恰好 1 秒。
        self.assertEqual(leap_epoch.tai_nanoseconds + NS,
                         after.tai_nanoseconds)
        # 闰秒那一刻沿用旧偏移 36，零点用新偏移 37。
        expect_leap = (civil_nanoseconds(2016, 12, 31, 0, 0, 0)
                       + (23 * 3600 + 59 * 60 + 60) * NS + 36 * NS)
        expect_zero = civil_nanoseconds(2017, 1, 1, 0, 0, 0) + 37 * NS
        self.assertEqual(leap_epoch.tai_nanoseconds, expect_leap)
        self.assertEqual(after.tai_nanoseconds, expect_zero)
        self.assertEqual(leap_epoch.time_system, "UTC")

    def test_gps_is_tai_minus_19s(self):
        gps = gps_to_tai_ns(2026, 10, 27, 0, 0, "  0.0000000")
        civil = civil_nanoseconds(2026, 10, 27, 0, 0, 0)
        self.assertEqual(gps, civil + 19 * NS)

    def test_unknown_leap_outside_frozen_table(self):
        # 表首个生效 2015-07-01：更早的日期必须报未知，不用本机时区猜。
        with self.assertRaises(LeapSecondUnknown):
            self.table.utc_to_tai_ns(2010, 1, 1, 0, 0, "  0.0000000")
        # valid_until 2027-01-01 当日及之后同样未知。
        with self.assertRaises(LeapSecondUnknown):
            self.table.utc_to_tai_ns(2027, 1, 1, 0, 0, "  0.0000000")

    def test_utc_without_table_marks_unknown_not_raises(self):
        _, epochs = parse_file(DATA / "utc-leap.rnx", None)
        self.assertTrue(all(e.time_axis_status == "unknown_leap"
                            and e.tai_nanoseconds is None for e in epochs))


class ClockOffsetTests(unittest.TestCase):
    def test_clock_offset_archived_separately(self):
        rnx = "\n".join([
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G    1 C1C", "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     1", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            "> 2026 10 27 00 00  0.0000000  0  1   1.234567890123",
            obs_line("G01", ["      10.0000"]),
        ]) + "\n"
        header, epochs = parse_bytes(rnx.encode("utf-8"))
        self.assertEqual(header.clock_offset_applied, 1)
        epoch = epochs[0]
        # 钟偏单列归档，观测值与历元时刻不被再次校正。
        self.assertAlmostEqual(epoch.clock_offset, 1.234567890123)
        self.assertAlmostEqual(epoch.satellites[0].slots[0].value, 10.0)
        self.assertEqual(epoch.second_text.strip(), "0.0000000")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("normal.json")
        self.path = DATA / "observations.rnx"

    def test_resume_does_not_apply_old_header_to_new_epoch(self):
        with ObservationCollector(self.path, leap(self.m)) as collector:
            collector.read_header()
            it = collector.epochs()
            first = next(it)           # flag 0，版本 0
            update = next(it)          # flag 4，升到版本 1
            self.assertEqual(update.catalog_version, 1)
            snapshot = collector.checkpoint()  # 完整历元边界
            self.assertEqual(snapshot.version, 1)
            self.assertEqual(snapshot.catalogs["G"],
                             ("C1C", "L1C", "S1C", "D1C"))

        # 模拟重启：新采集器从快照恢复，旧头版本不能套给新历元。
        with ObservationCollector(self.path, leap(self.m)) as restarted:
            restarted.read_header()
            restarted.resume(snapshot)
            remaining = list(restarted.epochs())
        self.assertEqual(len(remaining), 2)       # flag 0 与 flag 6
        self.assertTrue(all(e.catalog_version == 1 for e in remaining))
        self.assertEqual(len(remaining[0].satellites[0].slots), 4)

    def test_checkpoint_byte_offset_resumes_midstream(self):
        with ObservationCollector(self.path, leap(self.m)) as collector:
            collector.read_header()
            epoch = next(collector.epochs())
            snap = collector.checkpoint()
        raw = self.path.read_bytes()
        # 快照偏移正好指向下一个历元行的 '>'。
        self.assertEqual(raw[snap.next_byte_offset: snap.next_byte_offset + 1], b">")
        self.assertGreater(snap.next_byte_offset, epoch.byte_start)


class MultiSystemAndFirstObsTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("normal.json")
        self.table = leap(self.m)

    def test_each_system_uses_its_own_catalog_slots(self):
        rows = [
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G    2 C1C L1C", "SYS / # / OBS TYPES "),
            head_line("E    3 C1X L1X S1X", "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            "> 2026 10 27 00 00  0.0000000  0  2",
            obs_line("G01", ["      10.0000", "      20.0000"]),
            obs_line("E02", ["      30.0000", "      40.0000", "      50.0000"]),
        ]
        header, epochs = parse_bytes(("\n".join(rows) + "\n").encode(), self.table)
        self.assertEqual(header.catalogs, {
            "G": ["C1C", "L1C"], "E": ["C1X", "L1X", "S1X"]})
        g01, e02 = epochs[0].satellites
        self.assertEqual([o.code for o in g01.slots], ["C1C", "L1C"])
        self.assertEqual(g01.physical_slots, 2)
        self.assertEqual([o.code for o in e02.slots], ["C1X", "L1X", "S1X"])
        self.assertEqual(e02.physical_slots, 3)

    def test_time_of_first_obs_mismatch_reported_not_forced(self):
        rows = [
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G    1 C1C", "SYS / # / OBS TYPES "),
            # 头部声称首历元在 10 月 26 日。
            head_line("  2026    10    26     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            # 正文首历元实际在 10 月 27 日。
            "> 2026 10 27 00 00  0.0000000  0  1",
            obs_line("G01", ["      10.0000"]),
        ]
        _, epochs = parse_bytes(("\n".join(rows) + "\n").encode(), self.table)
        mismatch = epochs[0].first_obs_mismatch
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["header_time_of_first_obs"]["day"], 26)
        self.assertEqual(mismatch["first_epoch"]["day"], 27)
        # 后续历元不再重复报告。
        self.assertTrue(all(e.first_obs_mismatch is None for e in epochs[1:]))

    def test_matching_first_obs_has_no_mismatch(self):
        rows = [
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G    1 C1C", "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            "> 2026 10 27 00 00  0.0000000  0  1",
            obs_line("G01", ["      10.0000"]),
        ]
        _, epochs = parse_bytes(("\n".join(rows) + "\n").encode(), self.table)
        self.assertIsNone(epochs[0].first_obs_mismatch)


class StructuralRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.m = load_manifest("normal.json")
        self.table = leap(self.m)

    def _minimal(self, epoch_line: str, sat_lines: list[str] | None = None,
                 catalog: str = "G    1 C1C") -> bytes:
        rows = [
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line(catalog, "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            epoch_line,
        ]
        rows.extend(sat_lines or [])
        return ("\n".join(rows) + "\n").encode("utf-8")

    def test_non_utf8_byte_rejected(self):
        data = self._minimal("> 2026 10 27 00 00  0.0000000  0  1",
                             [obs_line("G01", ["      10.0000"])])
        bad = data[:200] + b"\xff\xfe" + data[200:]
        with self.assertRaises(RinexError):
            parse_bytes(bad, self.table)

    def test_missing_satellite_line_marks_incomplete(self):
        # 声明 1 颗卫星但卫星行整个缺失（EOF）。
        data = self._minimal("> 2026 10 27 00 00  0.0000000  0  1")
        _, epochs = parse_bytes(data, self.table)
        self.assertEqual(len(epochs), 1)
        self.assertFalse(epochs[0].complete)
        self.assertIn("文件结束", epochs[0].incomplete_reason)
        self.assertEqual(epochs[0].satellites, ())

    def test_second_field_beyond_nanosecond_precision_rejected(self):
        self.assertEqual(_seconds_to_integer_nanoseconds("  0.000000001", 1), 1)
        with self.assertRaises(RinexError):
            _seconds_to_integer_nanoseconds("0.0000000001", 1)

    def test_flag4_with_continuation_lines_updates_atomically(self):
        first = head_line(
            "G   15 C1C L1C S1C D1C C1W L1W S1W C2W L2W S2W C2L L2L S2L",
            "SYS / # / OBS TYPES ")
        cont = head_line("        C5Q L5Q", "SYS / # / OBS TYPES ")
        rows = [
            head_line("     3.04           OBSERVATION DATA    M",
                      "RINEX VERSION / TYPE"),
            head_line("G    2 C1C L1C", "SYS / # / OBS TYPES "),
            head_line("  2026    10    27     0     0    0.0000000     GPS",
                      "TIME OF FIRST OBS   "),
            head_line("     0", "RCV CLOCK OFFS APPL "),
            head_line("", "END OF HEADER       "),
            "> 2026 10 27 00 00  0.0000000  4  2",
            first, cont,
            "> 2026 10 27 00 00  1.0000000  0  1",
            obs_line("G01", ["      10.0000"] * 15),
        ]
        _, epochs = parse_bytes(("\n".join(rows) + "\n").encode(), self.table)
        self.assertEqual(len(epochs[0].header_update["catalogs"]["G"]), 15)
        self.assertEqual(epochs[1].catalog_version, 1)
        self.assertEqual(len(epochs[1].satellites[0].slots), 15)

    def test_zero_satellite_epoch_is_complete_empty(self):
        data = self._minimal("> 2026 10 27 00 00  0.0000000  0  0")
        _, epochs = parse_bytes(data, self.table)
        self.assertTrue(epochs[0].complete)
        self.assertEqual(epochs[0].satellites, ())

    def test_extra_slots_beyond_catalog_mark_incomplete(self):
        # 目录只有 1 个观测码，但物理行给了 2 个槽：不能静默吞掉多余槽。
        data = self._minimal(
            "> 2026 10 27 00 00  0.0000000  0  1",
            [obs_line("G01", ["      10.0000", "      20.0000"])])
        _, epochs = parse_bytes(data, self.table)
        self.assertFalse(epochs[0].complete)
        self.assertIn("物理槽 2 与目录 1", epochs[0].incomplete_reason)
        # 仅按目录保留 1 个槽，多余槽不产生伪观测码。
        self.assertEqual(
            [o.code for o in epochs[0].satellites[0].slots], ["C1C"])


if __name__ == "__main__":
    unittest.main()
