# =========================================================
# Stage 2 + Stage 3 — targeted check on lap-alignment-warning races
# Races: 2021 Round 3 (Portuguese), 2021 Round 8 (Styrian), 2022 Round 2 (Saudi Arabian)
# These three hit the "Ignoring late data" / "Failed to align laps for
# drivers" / "timing integrity error" / "No lap data for driver X" warning
# family flagged in Data Pipeline Plan.md as the one risk category not yet
# exercised by the two 2021 validation races already confirmed clean.
# Same fixed Stage 2/3 logic as before (includes the stream_data Driver
# car-number -> abbreviation remap). Reads data/raw/, writes data/stitched/.
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
    "2021_03_portuguese_grand_prix_R",   # "Ignoring late data" + "Failed to align laps"
    "2021_08_styrian_grand_prix_R",      # "Failed to align laps" + "timing integrity error (might be a bug)"
    "2022_02_saudi_arabian_grand_prix_R",  # "No lap data for driver 22" + "all laps inaccurate" x2
]
GRID_FREQ = "5s"

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
    unmapped = stream_df["Driver"].isna().sum()
    if unmapped:
        print(f"    WARNING: {unmapped}/{len(stream_df)} stream_data rows had a car number "
              f"not found in this race's laps_data — dropping them")
        stream_df = stream_df.dropna(subset=["Driver"])
    return stream_df

def build_grid(laps_df, freq=GRID_FREQ):
    grids = []
    for driver, laps in laps_df.groupby("Driver"):
        start = laps["LapStartTime"].min()
        end = laps["Time"].max()
        if pd.isna(start) or pd.isna(end) or end <= start:
            print(f"    WARNING: skipping {driver}, bad bounds (start={start}, end={end})")
            continue
        times = pd.timedelta_range(start=start, end=end, freq=freq)
        grids.append(pd.DataFrame({"Driver": driver, "Time": times}))
    if not grids:
        raise RuntimeError("no drivers produced a usable grid at all")
    return pd.concat(grids, ignore_index=True).sort_values(["Driver", "Time"]).reset_index(drop=True)

def stitch_race(name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    laps_df = pd.read_parquet(RAW_DIR / f"{name}_laps.parquet")
    stream_df = pd.read_parquet(RAW_DIR / f"{name}_stream.parquet")
    car_df = pd.read_parquet(RAW_DIR / f"{name}_car.parquet")

    print(f"[load] laps: {len(laps_df)} rows, stream: {len(stream_df)} rows, car: {len(car_df)} rows")
    print(f"       drivers -> laps: {laps_df['Driver'].nunique()}, stream (raw): {stream_df['Driver'].nunique()}, car: {car_df['Driver'].nunique()}")

    # flag any driver present in car/stream but missing from laps entirely (the "No lap data
    # for driver X" scenario) - these drivers should just be silently absent from the grid,
    # not crash anything. Confirm that's actually what happens.
    laps_drivers = set(laps_df["Driver"].unique())
    car_drivers = set(car_df["Driver"].unique())
    missing_from_laps = car_drivers - laps_drivers
    if missing_from_laps:
        print(f"[check] driver(s) with car telemetry but NO laps_data at all: {missing_from_laps} "
              f"-> expected to be silently absent from the grid (build_grid groups by laps_df's Driver)")

    stream_df = fix_stream_driver_ids(stream_df, laps_df)
    print(f"       stream drivers after remap: {stream_df['Driver'].nunique()}")

    grid = build_grid(laps_df)
    print(f"[grid] {len(grid)} rows, {grid['Driver'].nunique()} drivers "
          f"(laps_data had {laps_df['Driver'].nunique()} drivers)")
    if grid["Driver"].nunique() != laps_df["Driver"].nunique():
        print("       NOTE: driver count mismatch between laps_data and the built grid - "
              "check the 'bad bounds' warnings above for why")

    span = grid.groupby("Driver")["Time"].agg(["min", "max", "count"])
    session_end = span["max"].max()
    early_cutoff = session_end - pd.Timedelta(minutes=5)
    retired = span[span["max"] < early_cutoff]
    if len(retired):
        print(f"[check] retirement-cutoff: {len(retired)} driver(s) whose grid ends >5min before session end:")
        print(retired)
    else:
        print("[check] retirement-cutoff: no driver's grid ends notably early")

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

    print(f"[stitch] final table: {len(stitched)} rows, {len(stitched.columns)} columns")

    for col in ["gap_ahead", "gap_leader", "Position", "Speed"]:
        n_null = stitched[col].isna().sum()
        pct = n_null / len(stitched) * 100
        flag = "  <-- INVESTIGATE" if pct > 5 else ""
        print(f"[check] {col} NaN: {n_null}/{len(stitched)} ({pct:.1f}%){flag}")

    # check for any Time ordering / duplicate-timestamp weirdness a lap-alignment
    # problem could plausibly produce
    dupe_check = stitched.groupby(["Driver", "Time"]).size()
    dupes = dupe_check[dupe_check > 1]
    if len(dupes):
        print(f"[check] WARNING: {len(dupes)} (Driver, Time) combinations appear more than once in the grid - unexpected")
    else:
        print("[check] no duplicate (Driver, Time) grid rows - OK")

    # time-reference spot check on the driver most likely to be affected, if known
    sample_driver = laps_df["Driver"].iloc[0]
    driver_laps = laps_df[laps_df["Driver"] == sample_driver].dropna(subset=["LapStartTime", "Time"]).reset_index(drop=True)
    if len(driver_laps):
        sample_lap = driver_laps.iloc[len(driver_laps) // 2]
        lap_start, lap_end = sample_lap["LapStartTime"], sample_lap["Time"]
        window = stitched[(stitched["Driver"] == sample_driver) & (stitched["Time"] >= lap_start) & (stitched["Time"] <= lap_end)]
        print(f"[check] time-reference spot check - {sample_driver}, lap {sample_lap['LapNumber']}, {lap_start} -> {lap_end}:")
        print(window[["Time", "Speed", "DRS", "gap_ahead"]].to_string(index=False))

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
