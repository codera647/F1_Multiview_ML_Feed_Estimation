# =========================================================
# Stage 2 + Stage 3 — FULL SCALE-UP
# Runs the validated stitching pipeline (grid + merge_asof, with the
# stream_data car-number->abbreviation fix) across every race found in
# data/raw/, skipping anything already in data/stitched/.
#
# Race list is DISCOVERED from data/raw/, not hardcoded — this means the
# 6 sessions excluded from the training scope (2023 Round 2 skip, plus
# Rounds 3/4R/4S/5/6 never saved) are automatically absent and simply
# never attempted, no exclusion list needed. Confirmed on 5 races so far
# (2021 Bahrain, Imola, Portugal, Styria; 2022 Saudi Arabia) covering both
# the normal case and the lap-alignment-warning edge case.
# =========================================================

import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

print(f"python : {sys.executable}")
print(f"pandas : {pd.__version__}")
if int(pd.__version__.split(".")[0]) >= 3:
    raise RuntimeError(f"pandas {pd.__version__} >= 3.0, wrong kernel?")

RAW_DIR = Path("data") / "raw"
STITCHED_DIR = Path("data") / "stitched"
STITCHED_DIR.mkdir(parents=True, exist_ok=True)

GRID_FREQ = "5s"
GAP_NULL_WARN_PCT = 5.0       # flag in the summary if gap_ahead null% exceeds this
CONSECUTIVE_FAILURE_LIMIT = 3  # circuit breaker, matches the Stage 1 loader's pattern

# ---------- discover races from data/raw/ ----------
def discover_races(raw_dir):
    """A race is only included if it has laps + stream + car files (the three
    this pipeline actually reads). Missing/partial races (e.g. the old-format
    2023_01_bahrain_R duplicate with only 2 files) are silently skipped."""
    stems = set()
    for p in raw_dir.glob("*_laps.parquet"):
        stems.add(p.name[: -len("_laps.parquet")])
    races = []
    incomplete = []
    for stem in sorted(stems):
        needed = [raw_dir / f"{stem}_laps.parquet", raw_dir / f"{stem}_stream.parquet", raw_dir / f"{stem}_car.parquet"]
        if all(f.exists() for f in needed):
            races.append(stem)
        else:
            incomplete.append(stem)
    return races, incomplete

# ---------- same fixed Stage 2/3 logic, validated on 5 races ----------
def parse_gap_value(raw, position):
    if pd.isna(raw):
        return np.nan, "missing"
    s = str(raw).strip()
    if re.fullmatch(r"[+-]\d+\.?\d*", s):
        return float(s), "real"
    if re.fullmatch(r"LAP \d+", s):
        return (0.0 if position == 1 else np.nan), "lap_marker"
    if re.fullmatch(r"\d+ L", s):
        return np.nan, "lapped"
    if re.fullmatch(r"\d+L", s):
        return np.nan, "malformed"
    return np.nan, "malformed"

def clean_gap_column(df, raw_col, out_col):
    parsed = df.apply(lambda r: parse_gap_value(r[raw_col], r["Position"]), axis=1)
    df[out_col] = [p[0] for p in parsed]
    df[f"{out_col}_kind"] = [p[1] for p in parsed]
    df[f"{out_col}_is_lapped"] = df[f"{out_col}_kind"] == "lapped"
    df[out_col] = df.groupby("Driver")[out_col].ffill()
    return df

def fix_stream_driver_ids(stream_df, laps_df):
    num_to_abbr = dict(laps_df[["DriverNumber", "Driver"]].drop_duplicates().values)
    stream_df = stream_df.copy()
    stream_df["Driver"] = stream_df["Driver"].map(num_to_abbr)
    stream_df = stream_df.dropna(subset=["Driver"])
    return stream_df

def build_grid(laps_df, freq=GRID_FREQ):
    grids = []
    for driver, laps in laps_df.groupby("Driver"):
        start = laps["LapStartTime"].min()
        end = laps["Time"].max()
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        times = pd.timedelta_range(start=start, end=end, freq=freq)
        grids.append(pd.DataFrame({"Driver": driver, "Time": times}))
    if not grids:
        raise RuntimeError("no drivers produced a usable grid at all")
    return pd.concat(grids, ignore_index=True).sort_values(["Driver", "Time"]).reset_index(drop=True)

def stitch_race(name):
    laps_df = pd.read_parquet(RAW_DIR / f"{name}_laps.parquet")
    stream_df = pd.read_parquet(RAW_DIR / f"{name}_stream.parquet")
    car_df = pd.read_parquet(RAW_DIR / f"{name}_car.parquet")

    stream_df = fix_stream_driver_ids(stream_df, laps_df)
    grid = build_grid(laps_df)

    stream_df = stream_df.sort_values(["Driver", "Time"]).reset_index(drop=True)
    stream_df = clean_gap_column(stream_df, "IntervalToPositionAhead", "gap_ahead")
    stream_df = clean_gap_column(stream_df, "GapToLeader", "gap_leader")

    stream_cols = ["Driver", "Time", "Position", "gap_ahead", "gap_ahead_is_lapped", "gap_leader", "gap_leader_is_lapped"]
    stitched = pd.merge_asof(grid.sort_values("Time"), stream_df[stream_cols].sort_values("Time"),
                              on="Time", by="Driver", direction="backward")

    car_cols = ["Driver", "Time", "Speed", "DRS"]
    stitched = pd.merge_asof(stitched.sort_values("Time"), car_df[car_cols].sort_values("Time"),
                              on="Time", by="Driver", direction="backward")

    laps_join = laps_df.rename(columns={"Time": "_LapEndTime", "LapStartTime": "Time"})
    laps_cols = ["Driver", "Time", "LapNumber", "Compound", "TyreLife", "Stint", "PitInTime", "PitOutTime", "TrackStatus"]
    stitched = pd.merge_asof(stitched.sort_values("Time"), laps_join[laps_cols].sort_values("Time"),
                              on="Time", by="Driver", direction="backward")

    return stitched

# ---------- run ----------
races, incomplete = discover_races(RAW_DIR)
print(f"discovered {len(races)} races with a complete raw triple (laps+stream+car)")
if incomplete:
    print(f"skipping {len(incomplete)} incomplete race(s), missing one or more of laps/stream/car: {incomplete}")

already_done = {p.stem for p in STITCHED_DIR.glob("*.parquet")}
todo = [r for r in races if r not in already_done]
print(f"{len(already_done)} already stitched, {len(todo)} remaining to process\n")

summary = {"ok": [], "failed": [], "high_null": []}
consecutive_failures = 0

for i, name in enumerate(todo, 1):
    t0 = time.time()
    try:
        stitched = stitch_race(name)
        out_path = STITCHED_DIR / f"{name}.parquet"
        stitched.to_parquet(out_path)
        dt = time.time() - t0
        gap_null_pct = stitched["gap_ahead"].isna().mean() * 100
        n_rows = len(stitched)
        n_drivers = stitched["Driver"].nunique()
        flag = ""
        if gap_null_pct > GAP_NULL_WARN_PCT:
            flag = f"  <-- gap_ahead null {gap_null_pct:.1f}%, investigate"
            summary["high_null"].append((name, gap_null_pct))
        print(f"[{i}/{len(todo)}] OK   {name}  ({n_rows} rows, {n_drivers} drivers, {dt:.1f}s){flag}")
        summary["ok"].append(name)
        consecutive_failures = 0
    except Exception as e:
        dt = time.time() - t0
        print(f"[{i}/{len(todo)}] FAIL {name}  ({dt:.1f}s)")
        traceback.print_exc()
        summary["failed"].append((name, str(e)))
        consecutive_failures += 1
        if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            print(f"\nSTOPPING: {CONSECUTIVE_FAILURE_LIMIT} consecutive failures — likely an environment "
                  f"problem, not per-race data issues. Fix and rerun; already-stitched races are skipped "
                  f"automatically next time.")
            break

print("\n" + "=" * 70)
print(f"Done. {len(summary['ok'])} succeeded, {len(summary['failed'])} failed, "
      f"{len(summary['high_null'])} flagged for high gap_ahead null%.")
if summary["failed"]:
    print("\nFailed races:")
    for name, err in summary["failed"]:
        print(f"  {name}: {err}")
if summary["high_null"]:
    print("\nHigh null% races (worth a manual look before trusting):")
    for name, pct in summary["high_null"]:
        print(f"  {name}: {pct:.1f}%")
print("=" * 70)
