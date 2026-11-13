"""RINEX 3.04 观测文件采集器。

只读取 UTF-8 兼容 ASCII 的 RINEX *观测* 文件：
不读导航电文、不做定位求解、不把接收机钟偏再次应用到观测值或历元时刻。

设计要点（与 docs/data-contract.md 对应）：

* 文件头按固定列解释，``SYS / # / OBS TYPES`` 支持头部续行；
* 卫星观测行不以 80 列截断，按该系统目录声明的 16 字符槽定长读取，
  绝不按空白 split（缺测槽是空白，split 会让后续槽移位）；
* 观测值、LLI、SSI 分开保留，空槽为 ``None`` 而不是 0；
* 数值只接受严格数值语法，D/E 指数等价处理，非法字符直接报错；
* 历元 flag 区分观测（0/1）、事件（2/3/5）、头更新（4）、周跳（6）；
* flag 4 的头更新先在候选目录上整批验证，成功后才原子替换生效目录；
* 时间保留原时间系统与秒字段（含显式闰秒），GPS/UTC 到 TAI 整数纳秒轴的
  转换只依赖调用方传入的冻结闰秒表，表覆盖不到的日期报未知，不碰本机时区。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

# 3.04 头部各记录均为 80 列；标签在第 61-80 列（0 基 60:80）。
HEADER_WIDTH = 80
LABEL_SLICE = slice(60, 80)
# 卫星标识 3 列 + 每槽（F14.4 数值 14 列 + LLI 1 列 + SSI 1 列）。
SAT_ID_SLICE = slice(0, 3)
SLOT_WIDTH = 16
VALUE_SLICE = slice(0, 14)
LLI_INDEX = 14
SSI_INDEX = 15
# 历元行固定列（0 基半开区间），按 RINEX 3.04 正文紧凑历元格式实测标定。
EPOCH_SLICES = {
    "year": slice(2, 6),
    "month": slice(7, 9),
    "day": slice(10, 12),
    "hour": slice(13, 15),
    "minute": slice(16, 18),
    "second": slice(18, 29),   # F11 宽字段，可能是 60.x（显式正闰秒）
    "flag": slice(29, 32),
    "sat_count": slice(32, 35),
    "clock_offset": slice(35, 50),  # F15.12，可全空白
}
# RINEX 3 观测文件允许的卫星系统首字母。
OBSERVATION_SYSTEMS = frozenset("GRESCJI")
# flag 语义：0/1 观测；2/3/5 声明数量的事件记录；4 头更新；6 周跳。
OBSERVATION_FLAGS = frozenset({0, 1})
EVENT_FLAGS = frozenset({2, 3, 5})
HEADER_UPDATE_FLAG = 4
CYCLE_SLIP_FLAG = 6


class RinexError(ValueError):
    """RINEX 结构或数值语法错误，不静默吞掉。"""


class LeapSecondUnknown(RinexError):
    """UTC 时刻落在冻结闰秒表覆盖范围之外，无法确定 TAI 整数纳秒。"""


@dataclass(frozen=True)
class ObservationValue:
    """单个观测槽的三要素与原字节位置，空值为 None 而非 0。"""

    system: str
    code: str
    value: Optional[float]
    lli: Optional[int]
    ssi: Optional[int]
    line_number: int
    slot_index: int
    # 相对整个文件的字节半开区间，含 LLI/SSI 整槽 16 字符。
    byte_start: int
    byte_end: int
    value_byte_start: int
    value_byte_end: int
    catalog_version: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SatelliteRecord:
    """一颗卫星在某历元内的定长槽集合；物理截断时 slots 少于目录声明。"""

    satellite: str
    system: str
    line_number: int
    byte_start: int
    byte_end: int
    physical_slots: int
    slots: tuple[ObservationValue, ...]


@dataclass(frozen=True)
class EpochRecord:
    """一个历元行及其全部后续记录。

    flags 0/1 与 6 使用卫星记录；6 单独标记为周跳，不混入普通观测；
    flags 2/3/5 只保存事件行；flag 4 的结果记录在 header_update 上。
    """

    line_number: int
    byte_start: int
    byte_end: int
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second_text: str          # 原秒字段，保留 60.x 显式闰秒
    second: float
    time_system: str          # 该历元沿用文件头声明的时间系统
    flag: int
    declared_satellites: int
    clock_offset: Optional[float]
    catalog_version: int
    satellites: tuple[SatelliteRecord, ...] = ()
    event_lines: tuple[dict, ...] = ()
    cycle_slip: bool = False
    header_update: Optional[dict] = None
    complete: bool = True
    incomplete_reason: Optional[str] = None
    tai_nanoseconds: Optional[int] = None
    time_axis_status: str = "ok"  # ok / unknown_leap / not_applicable
    # 仅正文第一个历元非空：与头部 TIME OF FIRST OBS 的日期差异（不强制相等）。
    first_obs_mismatch: Optional[dict] = None

    def observations(self) -> Iterator[ObservationValue]:
        for sat in self.satellites:
            yield from sat.slots

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class RinexHeader:
    """文件头解析结果。"""

    version: float
    file_type: str
    satellite_system: str
    time_of_first_obs: dict
    clock_offset_applied: Optional[int]
    # system -> 观测码列表；版本 0 即文件头给出的初始目录。
    catalogs: dict[str, list[str]]
    raw_other: list[dict] = field(default_factory=list)
    end_line_number: int = 0


@dataclass(frozen=True)
class CatalogSnapshot:
    """可用于断点续传的不可变目录版本。"""

    version: int
    catalogs: dict[str, tuple[str, ...]]
    epoch_line_number: int
    # 下一历元行的文件字节偏移；checkpoint 只能在完整历元边界创建。
    next_byte_offset: int


# ---------------------------------------------------------------------------
# 基础数值/文本
# ---------------------------------------------------------------------------

def _strict_int(text: str, what: str, line_number: int) -> int:
    s = text.strip()
    if not s:
        raise RinexError(f"第{line_number}行 {what} 为空")
    try:
        return int(s)
    except ValueError as exc:
        raise RinexError(f"第{line_number}行 {what} 不是整数: {text!r}") from exc


def parse_observation_number(text: str, line_number: int, what: str = "观测值") -> float:
    """解析 F14.4 字段：全空白为缺测 None；D/E 指数等价，其余严格报错。"""

    if text == " " * len(text) or not text.strip():
        return None
    s = text.strip()
    # 显式拒绝浮点词法以外的内容（inf/nan/十六进制/下划线等）。
    lowered = s.lower()
    if lowered in {"nan", "inf", "-inf", "+inf", "infinity"}:
        raise RinexError(f"第{line_number}行 {what} 含非法数值词: {text!r}")
    # RINEX 历史上允许 D 作为指数标记，与 E 等价。
    candidate = s.replace("d", "e").replace("D", "E")
    try:
        return float(candidate)
    except ValueError as exc:
        raise RinexError(f"第{line_number}行 {what} 数值语法非法: {text!r}") from exc


def _parse_flag_field(text: str, line_number: int) -> Optional[int]:
    """LLI/SSI 单字符：空白为 None，仅接受单个数字。"""

    if text == " " or text == "":
        return None
    if len(text) != 1 or not text.isdigit():
        raise RinexError(f"第{line_number}行 LLI/SSI 非法: {text!r}")
    return int(text)


def _parse_clock_offset(text: str, line_number: int) -> Optional[float]:
    if not text.strip():
        return None
    return parse_observation_number(text, line_number, "接收机钟偏")


def _seconds_to_integer_nanoseconds(second_text: str, line_number: int) -> int:
    """把 F11.7 秒字段转成整数纳秒；超过纳秒精度直接报错，不做静默舍入。"""

    s = second_text.strip()
    if "." in s:
        whole, frac = s.split(".", 1)
        if len(frac) > 9:
            raise RinexError(f"第{line_number}行秒字段超过纳秒精度: {second_text!r}")
        frac_ns = int(frac.ljust(9, "0"))
    else:
        whole, frac_ns = s, 0
    return int(whole) * 1_000_000_000 + frac_ns


# ---------------------------------------------------------------------------
# 民用日期（支持 60 秒）到儒略日/整数纳秒
# ---------------------------------------------------------------------------

def _civil_to_days(year: int, month: int, day: int) -> int:
    """Fliegel 公式，把民用日期换成自 1970-01-01 的天数（纯整数）。"""

    y = year
    m = month
    # 1、2 月视作上一年的 13、14 月。
    a = (14 - m) // 12
    y += 4800 - a
    m += 12 * a - 3
    julian_day = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return julian_day - 2440588  # 1970-01-01 的儒略日


def civil_nanoseconds(year: int, month: int, day: int,
                      hour: int, minute: int, second_ns: int) -> int:
    """民用日历轴上的整数纳秒（秒字段可含闰秒 60）。"""

    days = _civil_to_days(year, month, day)
    return (days * 86400 + hour * 3600 + minute * 60) * 1_000_000_000 + second_ns


# ---------------------------------------------------------------------------
# 冻结闰秒表与时间轴转换
# ---------------------------------------------------------------------------

class LeapTable:
    """只按冻结表工作的 UTC<->TAI 转换，绝不查询本机时区。"""

    def __init__(self, entries: list[dict], valid_until: str):
        if not entries:
            raise RinexError("闰秒表为空")
        points = []
        for e in entries:
            eff = self._parse_iso(e["effective_utc"])
            points.append((eff, int(e["tai_minus_utc"])))
        points.sort(key=lambda x: x[0])
        # 生效时刻必须严格递增，偏移也应递增（正闰秒表）。
        for (d1, o1), (d2, o2) in zip(points, points[1:]):
            if d2 <= d1:
                raise RinexError("闰秒表生效时刻必须严格递增")
            if o2 <= o1:
                raise RinexError("闰秒表偏移必须递增")
        self.points = points
        self.valid_until = self._parse_iso(valid_until)
        if self.valid_until <= points[-1][0]:
            raise RinexError("leap_valid_until 必须晚于最后一个生效时刻")

    @staticmethod
    def _parse_iso(text: str) -> int:
        """解析表内 Z 结尾 ISO 时刻为 TAI 轴之外使用的民用纳秒（无闰秒歧义）。"""

        if not text.endswith("Z") or len(text) < 20:
            raise RinexError(f"闰秒表时刻必须为 ...Z 的 ISO 形式: {text!r}")
        date_part, time_part = text[:-1].split("T", 1)
        y, mo, d = (int(x) for x in date_part.split("-"))
        hh, mm, ss = time_part.split(":")
        sec_text = ss
        if "." in sec_text:
            whole, frac = sec_text.split(".", 1)
            if len(frac) > 9:
                raise RinexError("闰秒表时刻超过纳秒精度")
            sec_ns = int(whole) * 1_000_000_000 + int(frac.ljust(9, "0"))
        else:
            sec_ns = int(sec_text) * 1_000_000_000
        return civil_nanoseconds(y, mo, d, int(hh), int(mm), sec_ns)

    def offset_for_utc(self, year: int, month: int, day: int,
                       hour: int, minute: int, second_ns: int,
                       is_leap_second: bool) -> int:
        """返回该 UTC 民用时刻应采用的 TAI-UTC 偏移秒数。

        正闰秒（秒字段 60.x）沿用新 effective 之前一秒的旧偏移；
        表前日期与 valid_until 当日及之后报 LeapSecondUnknown。
        全程整数纳秒，不经过浮点。
        """

        # 闰秒在民用轴上“借住”当天 23:59:60；查找边界用当天 00:00。
        boundary_ns = civil_nanoseconds(
            year, month, day, hour, minute, 0 if is_leap_second else second_ns)
        first_eff, first_off = self.points[0]
        # 表覆盖不到：早于首个生效时刻。
        if boundary_ns < first_eff:
            raise LeapSecondUnknown(
                f"UTC {year:04d}-{month:02d}-{day:02d} 早于闰秒表首个生效时刻")
        offset = first_off
        for eff, off in self.points:
            # 闰秒本身仍属于旧偏移区间：它与 effective 同处一个民用日边界，
            # 但语义上是“额外的一秒”，只接受严格早于该边界的生效点。
            if eff < boundary_ns if is_leap_second else eff <= boundary_ns:
                offset = off
        # valid_until 当日及之后不转换为已知。
        if boundary_ns >= self.valid_until:
            raise LeapSecondUnknown(
                f"UTC {year:04d}-{month:02d}-{day:02d} 不早于 leap_valid_until")
        return offset

    def utc_to_tai_ns(self, year: int, month: int, day: int,
                      hour: int, minute: int, second_text: str) -> int:
        second_ns = _seconds_to_integer_nanoseconds(second_text, 0)
        whole_seconds = int(second_text.strip().split(".", 1)[0])
        is_leap = whole_seconds >= 60
        offset = self.offset_for_utc(year, month, day, hour, minute,
                                     second_ns, is_leap)
        if is_leap:
            # 闰秒：当天 00:00 基准 + 当天已走过的纳秒（含 60 秒）+ 旧偏移。
            base = civil_nanoseconds(year, month, day, 0, 0, 0)
            into_day_ns = (hour * 3600 + minute * 60) * 1_000_000_000 + second_ns
            return base + into_day_ns + offset * 1_000_000_000
        civil = civil_nanoseconds(year, month, day, hour, minute, second_ns)
        return civil + offset * 1_000_000_000


def gps_to_tai_ns(year: int, month: int, day: int,
                  hour: int, minute: int, second_text: str) -> int:
    """GPS 民用时刻到 TAI 整数纳秒：TAI = GPS + 19s（冻结常量）。"""

    second_ns = _seconds_to_integer_nanoseconds(second_text, 0)
    if int(second_text.strip().split(".", 1)[0]) >= 60:
        raise RinexError("GPS 时间不允许 60 秒闰秒字段")
    return civil_nanoseconds(year, month, day, hour, minute, second_ns) + 19 * 1_000_000_000


# ---------------------------------------------------------------------------
# 文件头
# ---------------------------------------------------------------------------

def _validate_codes(system: str, codes: list[str], line_number: int) -> None:
    if system not in OBSERVATION_SYSTEMS:
        raise RinexError(f"第{line_number}行未知卫星系统 {system!r}")
    if len(codes) != len(set(codes)):
        raise RinexError(f"第{line_number}行系统 {system} 观测码重复: {codes}")
    for code in codes:
        # RINEX 3 观测码 = 类型字母 + 频段号 + 跟踪码，恰 3 个 ASCII 字符。
        if len(code) != 3 or not code.isalnum() or not code[0].isalpha():
            raise RinexError(f"第{line_number}行观测码非法: {code!r}")


def _parse_sys_obs_line(line: str, line_number: int,
                        pending: Optional[str]) -> tuple[str, Optional[int],
                                                         list[str], Optional[str]]:
    system = line[0:1]
    is_continuation = system == " "
    if is_continuation:
        if pending is None:
            raise RinexError(f"第{line_number}行观测类型续行缺少所属系统")
        system = pending
        # 续行不重复总数（第 4-6 列为空白），返回 None 由调用方沿用首行声明。
        count: Optional[int] = None
        if line[3:6].strip():
            raise RinexError(f"第{line_number}行续行不应携带观测类型总数")
    elif system not in OBSERVATION_SYSTEMS:
        raise RinexError(f"第{line_number}行未知卫星系统 {system!r}")
    else:
        count = _strict_int(line[3:6], "观测类型总数", line_number)
    codes: list[str] = []
    # 观测码从第 8 列起，每 4 列一个，60 列内最多 13 个。
    for i in range(13):
        cell = line[7 + 4 * i:11 + 4 * i]
        code = cell.strip()
        if code:
            codes.append(code)
    if not codes:
        raise RinexError(f"第{line_number}行 SYS / # / OBS TYPES 无观测码")
    return system, count, codes, system


def parse_header(lines: list[tuple[int, str]]) -> RinexHeader:
    """解析 END OF HEADER 之前的固定列头部。"""

    version = file_type = sat_sys = None
    first_obs: Optional[dict] = None
    clock_applied: Optional[int] = None
    catalogs: dict[str, list[str]] = {}
    declared_counts: dict[str, int] = {}
    raw_other: list[dict] = []
    pending_system: Optional[str] = None
    end_line_number = 0

    for line_number, line in lines:
        if len(line) < HEADER_WIDTH:
            raise RinexError(f"第{line_number}行头部不足 80 列: 长度 {len(line)}")
        label = line[LABEL_SLICE].strip()
        if label == "RINEX VERSION / TYPE":
            version = parse_observation_number(line[0:9].strip() or " ", line_number, "版本")
            if version is None or abs(version - 3.04) > 1e-9:
                raise RinexError(f"第{line_number}行仅支持 RINEX 3.04，实际 {version!r}")
            # RINEX 3.04：第 21 列文件类型（0 基 20，O=观测）；
            # 卫星系统在第 41 列（0 基 40，空格=M 混合系统）。
            file_type = line[20:21]
            if file_type != "O":
                raise RinexError(f"第{line_number}行不是观测文件 (FILE/TYPE={file_type!r})")
            sat_sys = line[40:41]
            # 空格或 M 均表示混合系统文件；具体系统由各 SYS / # / OBS TYPES 给出。
            if sat_sys not in OBSERVATION_SYSTEMS and sat_sys not in (" ", "M"):
                raise RinexError(f"第{line_number}行系统标识非法: {sat_sys!r}")
        elif label == "SYS / # / OBS TYPES":
            system, count, codes, pending_system = _parse_sys_obs_line(
                line, line_number, pending_system)
            catalogs.setdefault(system, [])
            catalogs[system].extend(codes)
            # 总数只在每个系统的首行给出；续行 count 为 None。
            if count is not None:
                if system in declared_counts:
                    raise RinexError(
                        f"第{line_number}行系统 {system} 重复声明观测类型总数")
                declared_counts[system] = count
        elif label == "TIME OF FIRST OBS":
            # 头部宽字段（右对齐），与正文紧凑历元行的列不同。
            first_obs = {
                "year": _strict_int(line[2:6], "首历元年", line_number),
                "month": _strict_int(line[8:14], "首历元月", line_number),
                "day": _strict_int(line[14:20], "首历元日", line_number),
                "hour": _strict_int(line[20:26], "首历元时", line_number),
                "minute": _strict_int(line[26:33], "首历元分", line_number),
                "second_text": line[33:44],
                "time_system": line[48:51].strip(),
                "line_number": line_number,
            }
            if first_obs["time_system"] not in {"GPS", "UTC", "GAL", "QZS", "BDT", "GLO"}:
                raise RinexError(
                    f"第{line_number}行 TIME OF FIRST OBS 时间系统非法: "
                    f"{first_obs['time_system']!r}")
        elif label == "RCV CLOCK OFFS APPL":
            clock_applied = _strict_int(line[0:6], "钟偏应用标志", line_number)
            if clock_applied not in (0, 1):
                raise RinexError(f"第{line_number}行 RCV CLOCK OFFS APPL 只能是 0/1")
        elif label == "END OF HEADER":
            end_line_number = line_number
            break
        else:
            raw_other.append({"line_number": line_number, "label": label, "text": line[:60]})
    else:
        raise RinexError("缺少 END OF HEADER")

    if version is None:
        raise RinexError("缺少 RINEX VERSION / TYPE")
    if first_obs is None:
        raise RinexError("缺少 TIME OF FIRST OBS")
    if not catalogs:
        raise RinexError("缺少 SYS / # / OBS TYPES")
    for system, codes in catalogs.items():
        _validate_codes(system, codes, end_line_number)
        if declared_counts[system] != len(codes):
            raise RinexError(
                f"系统 {system} 声明 {declared_counts[system]} 个观测类型，"
                f"实际读到 {len(codes)} 个（含续行）")
    return RinexHeader(
        version=version,
        file_type=file_type,
        satellite_system=sat_sys,
        time_of_first_obs=first_obs,
        clock_offset_applied=clock_applied,
        catalogs=catalogs,
        raw_other=raw_other,
        end_line_number=end_line_number,
    )


# ---------------------------------------------------------------------------
# flag 4 头更新（候选目录整批验证，原子替换）
# ---------------------------------------------------------------------------

def _build_candidate_catalog(current: dict[str, tuple[str, ...]],
                             update_lines: list[tuple[int, str]]) -> dict[str, tuple[str, ...]]:
    # 先在独立候选对象上累计；任何一行非法都抛错，调用方的生效目录不动。
    candidate = {sys_: tuple(codes) for sys_, codes in current.items()}
    pending: Optional[str] = None
    building: Optional[str] = None
    updated_systems: set[str] = set()
    for line_number, line in update_lines:
        if line[LABEL_SLICE].strip() != "SYS / # / OBS TYPES":
            raise RinexError(
                f"第{line_number}行 flag 4 只允许 SYS / # / OBS TYPES 头更新")
        system, count, codes, pending = _parse_sys_obs_line(line, line_number, pending)
        is_continuation = line[0:1] == " "
        if is_continuation:
            if building is None or building != system:
                raise RinexError(
                    f"第{line_number}行 flag 4 续行没有可接续的系统声明")
            existing = list(candidate[system]) + codes
        else:
            if system in updated_systems:
                raise RinexError(
                    f"第{line_number}行 flag 4 中系统 {system} 重复声明")
            updated_systems.add(system)
            building = system
            existing = list(codes)
        candidate[system] = tuple(existing)
    # 收尾：每个被更新系统的累计码数必须等于其首行声明总数。
    pending = None
    declared: dict[str, int] = {}
    for line_number, line in update_lines:
        system, count, _, pending = _parse_sys_obs_line(line, line_number, pending)
        if count is not None:
            declared[system] = count
    for system, total in declared.items():
        if total != len(candidate[system]):
            raise RinexError(
                f"flag 4 系统 {system} 声明 {total} 个类型，候选目录只有 "
                f"{len(candidate[system])} 个，更新整体作废")
    for system, codes in candidate.items():
        _validate_codes(system, list(codes), update_lines[-1][0])
    return candidate


# ---------------------------------------------------------------------------
# 主采集器
# ---------------------------------------------------------------------------

class ObservationCollector:
    """顺序采集 RINEX 3.04 观测文件，逐历元标注目录版本与事件上下文。

    构造接受文件路径，或通过 :meth:`from_bytes` 接受已做身份校验的字节流；
    字节偏移始终相对该 RINEX 逻辑文件的第 0 字节，便于还原原槽位置。
    """

    def __init__(self, path: str | Path, leap_table: Optional[LeapTable] = None,
                 max_epoch_satellites: int = 999):
        self.path = Path(path)
        self.leap_table = leap_table
        self.max_epoch_satellites = max_epoch_satellites
        self._f = self.path.open("rb")
        self._owns_stream = True
        self._source_name = str(self.path)
        self._line_number = 0
        self.header: Optional[RinexHeader] = None
        # 当前生效目录（不可变 tuple），版本号从 0 起，每次成功 flag4 +1。
        self._catalogs: dict[str, tuple[str, ...]] = {}
        self._catalog_version = 0
        self._body_epoch_seen = False

    @classmethod
    def from_bytes(cls, data: bytes, leap_table: Optional[LeapTable] = None,
                   max_epoch_satellites: int = 999,
                   source_name: str = "<bytes>") -> "ObservationCollector":
        """对已通过 sha256/长度校验的字节做采集，不落临时文件。"""

        import io
        collector = cls.__new__(cls)
        collector.leap_table = leap_table
        collector.max_epoch_satellites = max_epoch_satellites
        collector.path = None
        collector._f = io.BytesIO(data)
        collector._owns_stream = False
        collector._source_name = source_name
        collector._line_number = 0
        collector.header = None
        collector._catalogs = {}
        collector._catalog_version = 0
        collector._body_epoch_seen = False
        return collector

    @property
    def source_name(self) -> str:
        return self._source_name

    def close(self) -> None:
        if self._owns_stream:
            self._f.close()

    def __enter__(self) -> "ObservationCollector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- 行级读取 ----------------------------------------------------------

    def _read_line(self) -> Optional[tuple[int, str, int, int]]:
        start = self._f.tell()
        raw = self._f.readline()
        if not raw:
            return None
        self._line_number += 1
        end = self._f.tell()
        # UTF-8 兼容 ASCII：先按 UTF-8 解码，定长列均为 ASCII，拒绝非法字节。
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RinexError(f"第{self._line_number}行不是合法 UTF-8: {exc}") from exc
        if text.endswith("\n"):
            text = text[:-1]
        if text.endswith("\r"):
            text = text[:-1]
        return self._line_number, text, start, end

    def _require_line(self, context: str) -> tuple[int, str, int, int]:
        got = self._read_line()
        if got is None:
            raise RinexError(f"{context} 时文件提前结束")
        return got

    # -- 头部 --------------------------------------------------------------

    def read_header(self) -> RinexHeader:
        head_lines: list[tuple[int, str]] = []
        while True:
            got = self._read_line()
            if got is None:
                raise RinexError("文件在 END OF HEADER 前结束")
            line_number, text, start, end = got
            head_lines.append((line_number, text))
            if len(text) >= HEADER_WIDTH and text[LABEL_SLICE].strip() == "END OF HEADER":
                self.header = parse_header(head_lines)
                self._catalogs = {s: tuple(c) for s, c in self.header.catalogs.items()}
                self._catalog_version = 0
                return self.header

    # -- 时间轴 ------------------------------------------------------------

    def _time_axis(self, epoch: dict) -> tuple[Optional[int], str]:
        ts = self.header.time_of_first_obs["time_system"]
        if ts == "GPS":
            return gps_to_tai_ns(epoch["year"], epoch["month"], epoch["day"],
                                 epoch["hour"], epoch["minute"],
                                 epoch["second_text"]), "ok"
        if ts == "UTC":
            if self.leap_table is None:
                return None, "unknown_leap"
            try:
                ns = self.leap_table.utc_to_tai_ns(
                    epoch["year"], epoch["month"], epoch["day"],
                    epoch["hour"], epoch["minute"], epoch["second_text"])
            except LeapSecondUnknown:
                return None, "unknown_leap"
            return ns, "ok"
        # 其他时间系统（GAL/BDT/GLO/QZS）本次不定义共同轴转换。
        return None, "not_applicable"

    def _first_obs_mismatch(self, epoch: dict) -> Optional[dict]:
        """首个正文历元与头部 TIME OF FIRST OBS 的日历差异（只报告不强制）。"""

        if self._body_epoch_seen:
            return None
        self._body_epoch_seen = True
        tofo = self.header.time_of_first_obs
        actual = (epoch["year"], epoch["month"], epoch["day"],
                  epoch["hour"], epoch["minute"])
        declared = (tofo["year"], tofo["month"], tofo["day"],
                    tofo["hour"], tofo["minute"])
        if actual == declared:
            return None
        return {
            "header_time_of_first_obs": {
                "year": tofo["year"], "month": tofo["month"],
                "day": tofo["day"], "hour": tofo["hour"],
                "minute": tofo["minute"],
                "second_text": tofo["second_text"],
            },
            "first_epoch": {
                "year": epoch["year"], "month": epoch["month"],
                "day": epoch["day"], "hour": epoch["hour"],
                "minute": epoch["minute"],
                "second_text": epoch["second_text"],
            },
        }

    # -- 卫星观测行 --------------------------------------------------------

    def _read_satellite(self, expected_version: int,
                        line_number: int, text: str,
                        start: int, end: int) -> SatelliteRecord:
        if len(text) < 3:
            raise RinexError(f"第{line_number}行卫星标识不足 3 列")
        sat = text[SAT_ID_SLICE]
        system = sat[0:1]
        if system not in self._catalogs:
            raise RinexError(
                f"第{line_number}行卫星 {sat!r} 的系统 {system!r} 不在当前目录")
        if not sat[1:3].isdigit():
            raise RinexError(f"第{line_number}行卫星标识非法: {sat!r}")
        codes = self._catalogs[system]
        expected_slots = len(codes)
        body = text[3:]
        physical_slots = len(body) // SLOT_WIDTH
        # 不向下一卫星行借字符：物理行不够长就是截断。
        if len(body) % SLOT_WIDTH != 0:
            raise RinexError(
                f"第{line_number}行观测区长度 {len(body)} 不是 16 的整数倍")
        slots: list[ObservationValue] = []
        for i in range(min(physical_slots, expected_slots)):
            slot_start = 3 + i * SLOT_WIDTH
            slot = text[slot_start:slot_start + SLOT_WIDTH]
            value_text = slot[VALUE_SLICE]
            value = parse_observation_number(value_text, line_number,
                                             f"{sat} 第{i + 1}槽观测值")
            lli = _parse_flag_field(slot[LLI_INDEX], line_number)
            ssi = _parse_flag_field(slot[SSI_INDEX], line_number)
            slots.append(ObservationValue(
                system=system,
                code=codes[i],
                value=value,
                lli=lli,
                ssi=ssi,
                line_number=line_number,
                slot_index=i,
                byte_start=start + slot_start,
                byte_end=start + slot_start + SLOT_WIDTH,
                value_byte_start=start + slot_start,
                value_byte_end=start + slot_start + 14,
                catalog_version=expected_version,
            ))
        return SatelliteRecord(
            satellite=sat,
            system=system,
            line_number=line_number,
            byte_start=start,
            byte_end=end,
            physical_slots=physical_slots,
            slots=tuple(slots),
        )

    # -- 主循环 ------------------------------------------------------------

    def epochs(self) -> Iterator[EpochRecord]:
        if self.header is None:
            self.read_header()
        while True:
            got = self._read_line()
            if got is None:
                return
            line_number, text, start, end = got
            if not text:
                raise RinexError(f"第{line_number}行为空，无法解释")
            if text[0] != ">":
                raise RinexError(
                    f"第{line_number}行应以 > 开始历元，实际 {text[:1]!r}")
            epoch = self._parse_epoch_line(line_number, text)
            flag = epoch["flag"]
            version_before = self._catalog_version
            if flag in OBSERVATION_FLAGS or flag == CYCLE_SLIP_FLAG:
                yield self._collect_satellite_epoch(epoch, start,
                                                    version_before,
                                                    cycle_slip=flag == CYCLE_SLIP_FLAG)
            elif flag in EVENT_FLAGS:
                yield self._collect_event_epoch(epoch, start, version_before)
            elif flag == HEADER_UPDATE_FLAG:
                yield self._collect_header_update(epoch, start, version_before)
            else:
                raise RinexError(f"第{line_number}行未知历元 flag={flag}")

    def _parse_epoch_line(self, line_number: int, text: str) -> dict:
        if len(text) < 35:
            raise RinexError(f"第{line_number}行历元记录不足 35 列")
        second_text = text[EPOCH_SLICES["second"]]
        second = parse_observation_number(second_text.strip() or " ",
                                          line_number, "历元秒")
        if second is None:
            raise RinexError(f"第{line_number}行历元秒缺失")
        flag = _strict_int(text[EPOCH_SLICES["flag"]], "历元 flag", line_number)
        count = _strict_int(text[EPOCH_SLICES["sat_count"]], "记录数", line_number)
        if count < 0:
            raise RinexError(f"第{line_number}行记录数为负")
        if count > self.max_epoch_satellites:
            raise RinexError(f"第{line_number}行记录数 {count} 超过上限")
        clock = _parse_clock_offset(text[EPOCH_SLICES["clock_offset"]], line_number)
        epoch = {
            "line_number": line_number,
            "year": _strict_int(text[EPOCH_SLICES["year"]], "年", line_number),
            "month": _strict_int(text[EPOCH_SLICES["month"]], "月", line_number),
            "day": _strict_int(text[EPOCH_SLICES["day"]], "日", line_number),
            "hour": _strict_int(text[EPOCH_SLICES["hour"]], "时", line_number),
            "minute": _strict_int(text[EPOCH_SLICES["minute"]], "分", line_number),
            "second_text": second_text,
            "second": second,
            "flag": flag,
            "count": count,
            "clock_offset": clock,
        }
        ts = self.header.time_of_first_obs["time_system"]
        whole_sec = int(second_text.strip().split(".", 1)[0])
        if whole_sec >= 60 and ts != "UTC":
            raise RinexError(f"第{line_number}行 {ts} 时间不允许 60 秒字段")
        return epoch

    def _collect_satellite_epoch(self, epoch: dict, start: int,
                                 version: int, cycle_slip: bool) -> EpochRecord:
        declared = epoch["count"]
        satellites: list[SatelliteRecord] = []
        seen: set[str] = set()
        incomplete_reason = None
        for idx in range(declared):
            got = self._read_line()
            if got is None:
                incomplete_reason = (
                    f"声明 {declared} 颗卫星，读到第 {idx + 1} 颗时文件结束")
                break
            sat_line, sat_text, sat_start, sat_end = got
            record = self._read_satellite(version, sat_line, sat_text,
                                          sat_start, sat_end)
            if record.satellite in seen:
                raise RinexError(
                    f"第{sat_line}行历元内卫星 {record.satellite} 标识重复")
            seen.add(record.satellite)
            expected = len(self._catalogs[record.system])
            if record.physical_slots != expected:
                incomplete_reason = (
                    f"第{sat_line}行 {record.satellite} 物理槽 {record.physical_slots} "
                    f"与目录 {expected} 不一致")
            satellites.append(record)
        if incomplete_reason is None and len(satellites) != declared:
            incomplete_reason = f"声明 {declared} 颗卫星，实际 {len(satellites)} 行"
        tai_ns, axis_status = self._time_axis(epoch)
        return EpochRecord(
            line_number=epoch["line_number"],
            byte_start=start,
            byte_end=self._f.tell(),
            year=epoch["year"], month=epoch["month"], day=epoch["day"],
            hour=epoch["hour"], minute=epoch["minute"],
            second_text=epoch["second_text"], second=epoch["second"],
            time_system=self.header.time_of_first_obs["time_system"],
            flag=epoch["flag"], declared_satellites=declared,
            clock_offset=epoch["clock_offset"],
            catalog_version=version,
            satellites=tuple(satellites),
            cycle_slip=cycle_slip,
            complete=incomplete_reason is None,
            incomplete_reason=incomplete_reason,
            tai_nanoseconds=tai_ns,
            time_axis_status=axis_status,
            first_obs_mismatch=self._first_obs_mismatch(epoch),
        )

    def _collect_event_epoch(self, epoch: dict, start: int,
                             version: int) -> EpochRecord:
        # flag 2/3/5：后续 N 行是事件/注释，绝不能当卫星计数。
        event_lines: list[dict] = []
        for _ in range(epoch["count"]):
            ln, txt, s, e = self._require_line(
                f"第{epoch['line_number']}行事件记录")
            event_lines.append({
                "line_number": ln,
                "byte_start": s,
                "byte_end": e,
                "text": txt,
            })
        return EpochRecord(
            line_number=epoch["line_number"], byte_start=start,
            byte_end=self._f.tell(),
            year=epoch["year"], month=epoch["month"], day=epoch["day"],
            hour=epoch["hour"], minute=epoch["minute"],
            second_text=epoch["second_text"], second=epoch["second"],
            time_system=self.header.time_of_first_obs["time_system"],
            flag=epoch["flag"], declared_satellites=epoch["count"],
            clock_offset=epoch["clock_offset"], catalog_version=version,
            event_lines=tuple(event_lines),
            first_obs_mismatch=self._first_obs_mismatch(epoch),
        )

    def _collect_header_update(self, epoch: dict, start: int,
                               version: int) -> EpochRecord:
        update_lines: list[tuple[int, str]] = []
        raw_events: list[dict] = []
        for _ in range(epoch["count"]):
            ln, txt, s, e = self._require_line(
                f"第{epoch['line_number']}行头更新")
            update_lines.append((ln, txt))
            raw_events.append({"line_number": ln, "byte_start": s,
                               "byte_end": e, "text": txt})
        # 先在候选目录上整批验证；任何一行非法都抛错，生效目录保持旧版本。
        candidate = _build_candidate_catalog(self._catalogs, update_lines)
        new_version = version + 1
        self._catalogs = candidate
        self._catalog_version = new_version
        return EpochRecord(
            line_number=epoch["line_number"], byte_start=start,
            byte_end=self._f.tell(),
            year=epoch["year"], month=epoch["month"], day=epoch["day"],
            hour=epoch["hour"], minute=epoch["minute"],
            second_text=epoch["second_text"], second=epoch["second"],
            time_system=self.header.time_of_first_obs["time_system"],
            flag=HEADER_UPDATE_FLAG, declared_satellites=epoch["count"],
            clock_offset=epoch["clock_offset"], catalog_version=new_version,
            event_lines=tuple(raw_events),
            header_update={
                "from_version": version,
                "to_version": new_version,
                "catalogs": {s: list(c) for s, c in candidate.items()},
            },
            first_obs_mismatch=self._first_obs_mismatch(epoch),
        )

    # -- 断点 --------------------------------------------------------------

    def checkpoint(self) -> CatalogSnapshot:
        """只允许在完整历元边界调用，快照当前目录版本与下一历元偏移。"""

        return CatalogSnapshot(
            version=self._catalog_version,
            catalogs=dict(self._catalogs),
            epoch_line_number=self._line_number,
            next_byte_offset=self._f.tell(),
        )

    def resume(self, snapshot: CatalogSnapshot) -> None:
        """按快照恢复目录版本并定位到下一历元，旧头版本不会套给新历元。"""

        self._catalogs = {s: tuple(c) for s, c in snapshot.catalogs.items()}
        self._catalog_version = snapshot.version
        self._f.seek(snapshot.next_byte_offset)
        self._line_number = snapshot.epoch_line_number


def parse_file(path: str | Path, leap_table: Optional[LeapTable] = None,
               max_epoch_satellites: int = 999) -> tuple[RinexHeader, list[EpochRecord]]:
    """一次性采集整个文件，返回头部与全部历元（含不完整标记）。"""

    with ObservationCollector(path, leap_table, max_epoch_satellites) as collector:
        header = collector.read_header()
        records = list(collector.epochs())
    return header, records


def parse_bytes(data: bytes, leap_table: Optional[LeapTable] = None,
                max_epoch_satellites: int = 999,
                source_name: str = "<bytes>") -> tuple[RinexHeader, list[EpochRecord]]:
    """对已校验字节做一次性采集，语义与 :func:`parse_file` 相同。"""

    with ObservationCollector.from_bytes(
            data, leap_table, max_epoch_satellites, source_name) as collector:
        header = collector.read_header()
        records = list(collector.epochs())
    return header, records
