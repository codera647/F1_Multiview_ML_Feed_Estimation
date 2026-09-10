# =========================================================
# Stage 2 + Stage 3 validation (FIXED — see note below)
# Races: 2021 Round 1 (Bahrain), Round 2 (Emilia Romagna)
# Reads data/raw/, builds the per-driver 5s grid, aligns stream_data
# (with the gap-parsing rule applied BEFORE the join), telemetry, and
# lap context onto it via merge_asof. Includes sanity checks, not just
# the join itself. Saves stitched output to data/stitched/.
#
# FIX (2026-09-10): stream_data's Driver column holds raw car-number
# strings ('10', '44', ...), NOT 3-letter abbreviations like laps_data
# and car_data use ('GAS', 'HAM', ...). This caused merge_asof(by="Driver")
# to match nothing, so gap_ahead/gap_leader/Position came back 100% NaN
# in every row of both races. Fix: remap stream_df['Driver'] from car
# number -> abbreviation (using that race's own laps_df) immediately
# after loading, before anything else touches it.
# =========================================================

import re
import sys
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

RACES = [
    "2021_01_bahrain_grand_prix_R",
    "2021_02_emilia_romagna_grand_prix_R",
]
GRID_FREQ = "5s"

def parse_gap_value(raw, position):
    """Four-shape parsing rule for GapToLeader / IntervalToPositionAhead."""
    if pd.isna(raw):
        return np.nan, "missing"
    s = str(raw).strip()
    if re.fullmatch(r"[+-]\d+\.?\d*", s):               # shape 1: real gap
        return float(s), "real"
    if re.fullmatch(r"LAP \d+", s):                       # shape 2: lap-crossing marker
        return (0.0 if position == 1 else np.nan), "lap_marker"
    if re.fullmatch(r"\d+ L", s):                          # shape 3: genuine lapped status
        return np.nan, "lapped"
    if re.fullmatch(r"\d+L", s):                           # shape 4: malformed, session-end artifact
        return np.nan, "malformed"
    return np.nan, "malformed"

def clean_gap_column(df, raw_col, out_col):
    """Clean at the stream's own native timestamps, BEFORE merge_asof ever sees it."""
    parsed = df.apply(lambda r: parse_gap_value(r[raw_col], r["Position"]), axis=1)
    df[out_col] = [p[0] for p in parsed]
    df[f"{out_col}_kind"] = [p[1] for p in parsed]
    df[f"{out_col}_is_lapped"] = df[f"{out_col}_kind"] == "lapped"
    df[out_col] = df.groupby("Driver")[out_col].ffill()
    return df

def fix_stream_driver_ids(stream_df, laps_df):
    """FIX: stream_df['Driver'] is car-number strings; remap to the same
    3-letter abbreviation scheme laps_df/car_df use, via laps_df's own
    DriverNumber -> Driver mapping (safest: built fresh per race)."""
    num_to_abbr = dict(laps_df[["DriverNumber", "Driver"]].drop_duplicates().values)
    stream_df = stream_df.copy()
    stream_df["Driver"] = stream_df["Driver"].map(num_to_abbr)
    unmapped = stream_df["Driver"].isna().sum()
    if unmapped:
        print(f"    WARNING: {unmapped}/{len(stream_df)} stream_data rows had a car number "
              f"not found in this race's laps_data — dropping them")
        stream_df = stream_df.dropna(subset=["Driver"])
    return stream_df

def build_grid(laps_df, freq=GRID_FREQ):
    """Stage 2: one row per driver per 5s step, each driver's own start/end bounds."""
    grids = []
    for driver, laps in laps_df.groupby("Driver"):
        start = laps["LapStartTime"].min()
        end = laps["Time"].max()
        if pd.isna(start) or pd.isna(end) or end <= start:
            print(f"    WARNING: skipping {driver}, bad bounds (start={start}, end={end})")
            continue
        times = pd.timedelta_range(start=start, end=end, freq=freq)
        grids.append(pd.DataFrame({"Driver": driver, "Time": times}))
    return pd.concat(grids, ignore_index=True).sort_values(["Driver", "Time"]).reset_index(drop=True)

def stitch_race(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    laps_df = pd.read_parquet(RAW_DIR / f"{name}_laps.parquet")
    stream_df = pd.read_parquet(RAW_DIR / f"{name}_stream.parquet")
    car_df = pd.read_parquet(RAW_DIR / f"{name}_car.parquet")

    print(f"[load] laps: {len(laps_df)} rows, stream: {len(stream_df)} rows, car: {len(car_df)} rows")
    print(f"       drivers -> laps: {laps_df['Driver'].nunique()}, stream (raw, car numbers): {stream_df['Driver'].nunique()}, car: {car_df['Driver'].nunique()}")

    # ---- FIX: remap stream_data's Driver from car number -> abbreviation ----
    stream_df = fix_stream_driver_ids(stream_df, laps_df)
    print(f"       stream drivers after remap: {stream_df['Driver'].nunique()} ({sorted(stream_df['Driver'].unique())})")

    # ---- Stage 2 ----
    grid = build_grid(laps_df)
    print(f"[grid] {len(grid)} rows, {grid['Driver'].nunique()} drivers")

    span = grid.groupby("Driver")["Time"].agg(["min", "max", "count"])
    session_end = span["max"].max()
    early_cutoff = session_end - pd.Timedelta(minutes=5)
    retired = span[span["max"] < early_cutoff]
    if len(retired):
        print(f"[check] retirement-cutoff: {len(retired)} driver(s) whose grid ends >5min before session end:")
        print(retired)
    else:
        print("[check] retirement-cutoff: no driver's grid ends notably early - no obvious retirement here")

    # ---- Stage 3a: clean stream_data BEFORE aligning ----
    stream_df = stream_df.sort_values(["Driver", "Time"]).reset_index(drop=True)
    stream_df = clean_gap_column(stream_df, "IntervalToPositionAhead", "gap_ahead")
    stream_df = clean_gap_column(stream_df, "GapToLeader", "gap_leader")
    print(f"[parse] IntervalToPositionAhead shapes: {dict(stream_df['gap_ahead_kind'].value_counts())}")
    print(f"[parse] gap_ahead still NaN after forward-fill: {stream_df['gap_ahead'].isna().sum()} / {len(stream_df)} "
          f"(expected: each driver's first sample before any real reading arrives)")

    # ---- Stage 3b/c/d: merge_asof onto the grid ----
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

    print(f"[stitch] final table: {len(stitched)} rows, {len(stitched.columns)} columns")
    print(f"         columns: {list(stitched.columns)}")

    # sanity check: gap_ahead should now be populated, not 100% NaN
    gap_nan_pct = stitched["gap_ahead"].isna().mean() * 100
    print(f"[check] gap_ahead NaN in final stitched table: {stitched['gap_ahead'].isna().sum()}/{len(stitched)} ({gap_nan_pct:.1f}%)")
    if gap_nan_pct > 5:
        print("        WARNING: still unexpectedly high - investigate further")

    # sanity check: TrackStatus (1 = green/all clear, anything else = flag/SC/VSC/red)
    print(f"[check] TrackStatus value counts:\n{stitched['TrackStatus'].value_counts(dropna=False)}")

    # sanity check: time-reference alignment on one real lap
    sample_driver = laps_df["Driver"].iloc[0]
    driver_laps = laps_df[laps_df["Driver"] == sample_driver].dropna(subset=["LapStartTime", "Time"]).reset_index(drop=True)
    sample_lap = driver_laps.iloc[len(driver_laps) // 2]
    lap_start, lap_end = sample_lap["LapStartTime"], sample_lap["Time"]
    window = stitched[(stitched["Driver"] == sample_driver) & (stitched["Time"] >= lap_start) & (stitched["Time"] <= lap_end)]
    print(f"[check] time-reference spot check - {sample_driver}, lap {sample_lap['LapNumber']}, {lap_start} -> {lap_end}:")
    print(window[["Time", "Speed", "DRS", "gap_ahead"]].to_string(index=False))
    print("        (Speed should rise/fall like a real lap - not flat, random, or from a different part of the race)")

    out_path = STITCHED_DIR / f"{name}.parquet"
    stitched.to_parquet(out_path)
    print(f"[save] {out_path} ({out_path.stat().st_size:,} bytes)")
    return stitched

results = {}
for name in RACES:
    try:
        results[name] = stitch_race(name)
    except Exception:
        print(f"\nFAILED on {name}")
        traceback.print_exc()

print("\n" + "=" * 70)
print(f"Done. {len(results)}/{len(RACES)} races stitched successfully.")
print("=" * 70)
