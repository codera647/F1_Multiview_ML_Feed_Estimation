# =========================================================
# Stage 4 — DERIVED FEATURES
#
# Reads data/stitched/{name}.parquet (Stage 2/3 output: the per-driver
# 5-second grid with gap_ahead/gap_leader/Position/tyre/pit/DRS already
# merge_asof'd on) plus data/raw/{name}_laps.parquet and
# data/raw/{name}_stream.parquet (for sector times and the raw GapToLeader
# string, neither of which survived the Stage 2/3 column selection), and
# adds the derived feature columns decided in the 2026-09-11 discussion.
# Output: one parquet per race in data/features/, stitched columns
# untouched, new columns appended. data/stitched/ itself is never
# modified, same reasoning as keeping labeling a separate pass: a Stage 4
# feature-definition tweak should never require re-running Stage 2/3.
#
# Three open design decisions this script implements (locked in
# 2026-09-11):
#   1. Segments with zero real underlying gap data for a driver's whole
#      race (e.g. 2024 Monaco's HUL/MAG/OCO/PER, retired lap 1 during the
#      red flag) are FLAGGED (stale_gap_data), never dropped or imputed.
#   2. sector_time_vs_best compares a driver's most recently COMPLETED
#      sector time against that same driver's own best time for that
#      sector number so far in the race (lagging, leak-safe by
#      construction). It does not attempt to identify which sector an
#      arbitrary mid-sector grid timestamp falls inside.
#   3. laps_behind is kept as a real integer feature (0 when on the lead
#      lap, N when N laps down), re-derived from the raw GapToLeader "N L"
#      string since the Stage 2/3 pipeline only kept a boolean flag.
#      Raw X/Y/Z pos_data is NOT pulled in here — left out of the v1
#      feature set per the same discussion.
#
# Also fixes a naming mismatch caught while writing this: the plan called
# for gap_change_3s/5s/10s, but the grid itself is only 5-second
# resolution, so a literal 3-second diff isn't achievable without
# interpolation. Renamed to gap_change_5s/10s/15s = 1/2/3 grid-step
# backward differences.
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
FEATURES_DIR = Path("data") / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

GRID_FREQ_S = 5
PIT_PROXIMITY_S = 30           # window for pit_stop_nearby_30s
DRS_OPEN_CODES = [10, 12, 14]  # FastF1 convention: DRS actually open (vs. 8 = eligible, 0/1 = off)
CONSECUTIVE_FAILURE_LIMIT = 3


# ---------- reused, unchanged from Stage 2/3 (kept identical on purpose) ----------
def fix_stream_driver_ids(stream_df, laps_df):
    num_to_abbr = dict(laps_df[["DriverNumber", "Driver"]].drop_duplicates().values)
    stream_df = stream_df.copy()
    stream_df["Driver"] = stream_df["Driver"].map(num_to_abbr)
    stream_df = stream_df.dropna(subset=["Driver"])
    return stream_df


# ---------- new: re-parse raw GapToLeader for the "N L" lap count Stage 2/3 discarded ----------
def parse_laps_behind(raw):
    """Returns an integer laps-behind count, or np.nan to mean 'hold previous
    state' (a LAP N recalculation marker, or a malformed session-end value)."""
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    m = re.fullmatch(r"(\d+) L", s)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"[+-]\d+\.?\d*", s):
        return 0  # a real numeric gap means "not lapped" right now
    return np.nan  # "LAP N" marker or malformed one-off -> unknown, ffill through it


def compute_laps_behind(stream_df):
    """stream_df must already have Driver fixed to 3-letter abbreviations and
    be sorted by (Driver, Time). Returns a Series aligned to stream_df's index."""
    raw_n = stream_df["GapToLeader"].map(parse_laps_behind)
    laps_behind = raw_n.groupby(stream_df["Driver"]).ffill().fillna(0).astype(int)
    return laps_behind


# ---------- new: sector_time_vs_best (lagging, own-best-so-far) ----------
def build_sector_vs_best_events(laps_df):
    """Long-format table: one row per (Driver, completed sector), with the
    sector's own duration, its completion timestamp (session time, same
    units as the grid's Time column), and how it compared to that driver's
    own best time for that sector number *before* this one (so the value
    never uses this sector's own time to judge itself)."""
    frames = []
    for n in (1, 2, 3):
        dur_col, ts_col = f"Sector{n}Time", f"Sector{n}SessionTime"
        if dur_col not in laps_df.columns or ts_col not in laps_df.columns:
            continue
        sub = laps_df[["Driver", dur_col, ts_col]].dropna(subset=[dur_col, ts_col]).copy()
        sub = sub.rename(columns={dur_col: "SectorTime", ts_col: "Time"})
        sub["SectorNumber"] = n
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["Driver", "Time", "SectorNumber", "SectorTime",
                                      "sector_time_vs_best", "sector_number_last_completed"])

    events = pd.concat(frames, ignore_index=True)
    events["SectorTime_s"] = events["SectorTime"].dt.total_seconds()
    events = events.sort_values(["Driver", "SectorNumber", "Time"])

    # expanding min of this driver's own SectorTime for this sector number,
    # shifted by one so the comparison never includes the sector being judged
    best_so_far = (
        events.groupby(["Driver", "SectorNumber"])["SectorTime_s"]
        .apply(lambda s: s.shift(1).expanding().min())
    )
    events["sector_time_vs_best"] = events["SectorTime_s"] - best_so_far.values
    events["sector_number_last_completed"] = events["SectorNumber"]

    events = events.sort_values(["Driver", "Time"]).reset_index(drop=True)
    return events[["Driver", "Time", "sector_time_vs_best", "sector_number_last_completed"]]


# ---------- main per-race feature build ----------
def add_features(name):
    stitched = pd.read_parquet(STITCHED_DIR / f"{name}.parquet")
    laps_df = pd.read_parquet(RAW_DIR / f"{name}_laps.parquet")
    stream_df = pd.read_parquet(RAW_DIR / f"{name}_stream.parquet")

    stitched = stitched.sort_values(["Driver", "Time"]).reset_index(drop=True)

    # --- decision 1: flag, don't drop or impute ---
    stitched["stale_gap_data"] = stitched["gap_ahead"].isna()

    # --- gap_change_5s/10s/15s: 1/2/3 grid-step backward diffs of gap_ahead ---
    step_s = GRID_FREQ_S
    for n_steps, label in ((1, "5s"), (2, "10s"), (3, "15s")):
        stitched[f"gap_change_{label}"] = stitched.groupby("Driver")["gap_ahead"].diff(n_steps)

    # --- in_drs_zone ---
    stitched["in_drs_zone"] = stitched["DRS"].isin(DRS_OPEN_CODES)

    # --- pit_stop_nearby_30s ---
    dt_in = (stitched["Time"] - stitched["PitInTime"]).abs()
    dt_out = (stitched["Time"] - stitched["PitOutTime"]).abs()
    thresh = pd.Timedelta(seconds=PIT_PROXIMITY_S)
    stitched["pit_stop_nearby_30s"] = (dt_in <= thresh).fillna(False) | (dt_out <= thresh).fillna(False)

    # --- tyre_age: alias of TyreLife, named per the README feature reference ---
    stitched["tyre_age"] = stitched["TyreLife"]

    # --- laps_behind: re-derived from raw stream data (Stage 2/3 only kept a bool) ---
    stream_df = fix_stream_driver_ids(stream_df, laps_df)
    stream_df = stream_df.sort_values(["Driver", "Time"]).reset_index(drop=True)
    stream_df["laps_behind"] = compute_laps_behind(stream_df)
    lb_cols = ["Driver", "Time", "laps_behind"]
    stitched = pd.merge_asof(
        stitched.sort_values("Time"), stream_df[lb_cols].sort_values("Time"),
        on="Time", by="Driver", direction="backward",
    )
    stitched["laps_behind"] = stitched["laps_behind"].fillna(0).astype(int)

    # --- sector_time_vs_best: lagging, own-best-so-far ---
    sector_events = build_sector_vs_best_events(laps_df)
    stitched = pd.merge_asof(
        stitched.sort_values("Time"), sector_events.sort_values("Time"),
        on="Time", by="Driver", direction="backward",
    )

    stitched = stitched.sort_values(["Driver", "Time"]).reset_index(drop=True)
    return stitched


# ---------- run ----------
def discover_races():
    stems = sorted(p.stem for p in STITCHED_DIR.glob("*.parquet"))
    return stems


if __name__ == "__main__":
    races = discover_races()
    print(f"discovered {len(races)} stitched races in {STITCHED_DIR}")

    already_done = {p.stem for p in FEATURES_DIR.glob("*.parquet")}
    todo = [r for r in races if r not in already_done]
    print(f"{len(already_done)} already featurized, {len(todo)} remaining\n")

    summary = {"ok": [], "failed": []}
    consecutive_failures = 0

    for i, name in enumerate(todo, 1):
        t0 = time.time()
        try:
            out = add_features(name)
            out.to_parquet(FEATURES_DIR / f"{name}.parquet")
            dt = time.time() - t0
            stale_pct = out["stale_gap_data"].mean() * 100
            lapped_rows = (out["laps_behind"] > 0).sum()
            flag = f"  <-- stale_gap_data {stale_pct:.1f}%" if stale_pct > 5.0 else ""
            print(f"[{i}/{len(todo)}] OK   {name}  ({len(out)} rows, "
                  f"{lapped_rows} laps_behind>0, {dt:.1f}s){flag}")
            summary["ok"].append(name)
            consecutive_failures = 0
        except Exception as e:
            dt = time.time() - t0
            print(f"[{i}/{len(todo)}] FAIL {name}  ({dt:.1f}s)")
            traceback.print_exc()
            summary["failed"].append((name, str(e)))
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                print(f"\nSTOPPING: {CONSECUTIVE_FAILURE_LIMIT} consecutive failures — likely an "
                      f"environment problem, not per-race data issues. Fix and rerun; already-"
                      f"featurized races are skipped automatically next time.")
                break

    print("\n" + "=" * 70)
    print(f"Done. {len(summary['ok'])} succeeded, {len(summary['failed'])} failed.")
    if summary["failed"]:
        print("\nFailed races:")
        for name, err in summary["failed"]:
            print(f"  {name}: {err}")
    print("=" * 70)
