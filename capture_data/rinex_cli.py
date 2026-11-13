"""RINEX 3.04 观测采集命令行。

两种入口：

* ``rinex-manifest data/normal.json``：读清单（含冻结闰秒表与资产校验），
  采集清单内全部观测资产；
* ``rinex-file data/observations.rnx --leap data/normal.json``：直接采集单个
  RINEX 观测文件，闰秒表从清单借用；UTC 文件未给表时只标注 unknown_leap。

输出为 JSON 序列：头部、逐历元（含事件上下文与目录版本）、逐条观测，
缺测槽以 ``value: null`` 保留位置，不完整历元显式标记。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load, asset
from .rinex3 import (
    LeapTable,
    ObservationCollector,
    RinexError,
)


def _leap_table(manifest: dict) -> LeapTable | None:
    if "leap_table" not in manifest:
        return None
    return LeapTable(manifest["leap_table"], manifest["leap_valid_until"])


def _epoch_payload(epoch) -> dict:
    return {
        "line_number": epoch.line_number,
        "byte_range": [epoch.byte_start, epoch.byte_end],
        "time": {
            "year": epoch.year,
            "month": epoch.month,
            "day": epoch.day,
            "hour": epoch.hour,
            "minute": epoch.minute,
            "second_text": epoch.second_text,
            "time_system": epoch.time_system,
            "tai_nanoseconds": epoch.tai_nanoseconds,
            "time_axis_status": epoch.time_axis_status,
        },
        "flag": epoch.flag,
        "declared_records": epoch.declared_satellites,
        "receiver_clock_offset": epoch.clock_offset,
        "catalog_version": epoch.catalog_version,
        "cycle_slip": epoch.cycle_slip,
        "complete": epoch.complete,
        "incomplete_reason": epoch.incomplete_reason,
        "first_obs_mismatch": epoch.first_obs_mismatch,
        "header_update": epoch.header_update,
        "event_lines": list(epoch.event_lines),
        "satellites": [
            {
                "satellite": sat.satellite,
                "system": sat.system,
                "line_number": sat.line_number,
                "byte_range": [sat.byte_start, sat.byte_end],
                "physical_slots": sat.physical_slots,
                "observations": [obs.to_dict() for obs in sat.slots],
            }
            for sat in epoch.satellites
        ],
    }


def _payload_from_collector(collector: ObservationCollector, source: str) -> dict:
    header = collector.read_header()
    epochs = [_epoch_payload(e) for e in collector.epochs()]
    return {
        "source": source,
        "header": {
            "version": header.version,
            "file_type": header.file_type,
            "satellite_system": header.satellite_system,
            "time_of_first_obs": header.time_of_first_obs,
            "receiver_clock_offset_applied": header.clock_offset_applied,
            "catalogs": {s: list(c) for s, c in header.catalogs.items()},
            "other_header_lines": header.raw_other,
            "end_line_number": header.end_line_number,
        },
        "epochs": epochs,
    }


def collect_path(path: str | Path, leap: LeapTable | None,
                 max_epoch_satellites: int = 999) -> dict:
    with ObservationCollector(path, leap, max_epoch_satellites) as collector:
        return _payload_from_collector(collector, str(path))


def collect_bytes(data: bytes, leap: LeapTable | None, source: str,
                  max_epoch_satellites: int = 999) -> dict:
    with ObservationCollector.from_bytes(
            data, leap, max_epoch_satellites, source) as collector:
        return _payload_from_collector(collector, source)


def _run_manifest(args) -> dict:
    manifest, _ = load(args.manifest)
    if manifest.get("kind") != "rinex-observation":
        raise RinexError("清单 kind 不是 rinex-observation")
    root = Path(args.manifest).parent
    leap = _leap_table(manifest)
    limit = int(manifest.get("max_epoch_satellites", 999))
    files = []
    for ref in manifest["assets"]:
        # asset() 已完成路径越界、长度与 sha256 身份校验，字节直接进采集器。
        raw = asset(root, ref)
        files.append(collect_bytes(raw, leap, ref["path"], limit))
    return {"manifest": str(args.manifest), "files": files}


def _run_file(args) -> dict:
    leap = None
    if args.leap:
        manifest, _ = load(args.leap)
        leap = _leap_table(manifest)
    return collect_path(args.rinex, leap)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RINEX 3.04 观测采集")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("rinex-manifest", help="按清单采集全部观测资产")
    p_manifest.add_argument("manifest")
    p_manifest.set_defaults(func=_run_manifest)

    p_file = sub.add_parser("rinex-file", help="直接采集单个观测文件")
    p_file.add_argument("rinex")
    p_file.add_argument("--leap", help="借用其 leap_table 的清单 JSON")
    p_file.set_defaults(func=_run_file)

    args = parser.parse_args(argv)
    payload = args.func(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
