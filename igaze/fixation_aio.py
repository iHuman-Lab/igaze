"""AOI-aware fixation extraction for config-driven eye-tracking datasets."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd



from igaze.detectors import fixation_detection
import yaml
from dataclasses import dataclass

# --- Begin inlined replacements for removed _eyetracking_common.py ---
@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    task_id: str | None
    file_path: Path
    eye_df: pd.DataFrame
    game_df: pd.DataFrame
    trial_df: pd.DataFrame

def scale_coordinates(x: pd.Series, y: pd.Series, et_cfg: dict) -> tuple[pd.Series, pd.Series]:
    if str(et_cfg.get("coordinate_system", "pixels")).lower() != "normalized":
        return x, y
    display_cfg = et_cfg.get("display", {})
    return x * float(display_cfg.get("width_px", 1920)), y * float(display_cfg.get("height_px", 1080))

def nearest_index(series: pd.Series, value: float) -> int:
    return int((series - value).abs().idxmin())

def empty_trial_info() -> dict:
    return {"trial_id": None, "llm_model": None, "prompt_type": None, "llm_provider": None}

def assign_trial(trial_df: pd.DataFrame, midpoint_timestamp: float) -> dict:
    if trial_df.empty:
        return empty_trial_info()
    mask = (trial_df["trial_start"] <= midpoint_timestamp) & (trial_df["trial_end"] >= midpoint_timestamp)
    row = (
        trial_df.loc[mask].iloc[0]
        if mask.any()
        else trial_df.iloc[(trial_df["trial_start"] - midpoint_timestamp).abs().argmin()]
    )
    return {
        "trial_id": int(row["trial_id"]),
        "llm_model": row["llm_model"],
        "prompt_type": row["prompt_type"],
        "llm_provider": row["llm_provider"],
    }

def with_subject_metadata(df: pd.DataFrame, subject_id: str, task_id, file_path: Path) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "file", str(file_path))
    df.insert(0, "task_id", task_id)
    df.insert(0, "subject_id", subject_id)
    return df

def overall_duration_seconds(time) -> float:
    return float(time[-1] - time[0]) / 1000.0 if len(time) > 1 else 0.0

def summarize_fixations(subject_fixations_df: pd.DataFrame, trial_df: pd.DataFrame, total_time_seconds: float) -> pd.DataFrame:
    if subject_fixations_df.empty:
        if trial_df.empty:
            return pd.DataFrame([
                {"trial_id": None, "llm_model": None, "prompt_type": None, "llm_provider": None, "n_fixations": 0, "mean_fixation_duration": 0.0, "total_fixation_time": 0.0, "total_time": total_time_seconds}
            ])
        durations = trial_df[["trial_id", "llm_model", "prompt_type", "llm_provider"]].copy()
        durations["total_time"] = (trial_df["trial_end"] - trial_df["trial_start"]).astype(float)
        durations["n_fixations"] = 0
        durations["mean_fixation_duration"] = 0.0
        durations["total_fixation_time"] = 0.0
        return durations
    group_cols = ["trial_id", "llm_model", "prompt_type", "llm_provider"]
    summary_group = (
        subject_fixations_df.groupby(group_cols, dropna=False)
        .agg(
            n_fixations=("fixation_id", "count"),
            mean_fixation_duration=("duration", "mean"),
            total_fixation_time=("duration", "sum"),
        )
        .reset_index()
    )
    if trial_df.empty:
        summary_group["total_time"] = total_time_seconds
        return summary_group
    durations = trial_df[["trial_id", "llm_model", "prompt_type", "llm_provider"]].copy()
    durations["total_time"] = (trial_df["trial_end"] - trial_df["trial_start"]).astype(float)
    summary_group = summary_group.merge(durations, on=group_cols, how="left")
    summary_group["total_time"] = summary_group["total_time"].fillna(total_time_seconds)
    return summary_group
# --- End inlined replacements ---


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _save_frame(project_root: Path, et_cfg: dict, key: str, frame: pd.DataFrame) -> None:
    default_outputs = {
        "raw_fixations_csv": "outputs/eyetracking/fixations_long.csv",
        "summary_csv": "outputs/eyetracking/fixations_summary.csv",
        "raw_fixations_aio_csv": "outputs/eyetracking/fixations_long_aio.csv",
        "summary_aio_csv": "outputs/eyetracking/fixations_summary_aio.csv",
        "eyetracker_timeline_csv": "outputs/eyetracking/eyetracker_timeline.csv",
        "llm_periods_csv": "outputs/eyetracking/llm_periods.csv",
        "raw_saccades_csv": "outputs/eyetracking/saccades_long.csv",
        "saccades_summary_csv": "outputs/eyetracking/saccades_summary.csv",
    }
    out_path = et_cfg.get("output", {}).get(key, default_outputs.get(key))
    out = (project_root / out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)


def _get_aoi_name(fix_x: float, fix_y: float, aois: list[dict]) -> str | None:
    for aoi in aois:
        if aoi["x_min"] <= fix_x <= aoi["x_max"] and aoi["y_min"] <= fix_y <= aoi["y_max"]:
            return str(aoi["name"])
    return None


def _subject_fixation_aoi_frames(
    subject_record: SubjectRecord,
    et_cfg: dict,
    aois: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = et_cfg["columns"]
    time_series = pd.to_numeric(subject_record.eye_df[columns["time"]], errors="coerce")
    x_series, y_series = scale_coordinates(
        pd.to_numeric(subject_record.eye_df[columns["x"]], errors="coerce"),
        pd.to_numeric(subject_record.eye_df[columns["y"]], errors="coerce"),
        et_cfg,
    )
    _, end_fixations = fixation_detection(
        x_series.to_numpy(),
        y_series.to_numpy(),
        time_series.to_numpy(),
        missing=et_cfg.get("missing", 0.0),
        maxdist=et_cfg["maxdist"],
        mindur=et_cfg["mindur"],
    )
    has_lsl = "_lsl_timestamp" in subject_record.eye_df.columns
    lsl_series = subject_record.eye_df["_lsl_timestamp"] if has_lsl else pd.Series(dtype=float)
    rows = []
    for fixation_id, (start_time, end_time, duration, fix_x, fix_y) in enumerate(end_fixations, start=1):
        start_idx = nearest_index(time_series, start_time)
        end_idx = nearest_index(time_series, end_time)
        midpoint = float((lsl_series.iloc[start_idx] + lsl_series.iloc[end_idx]) / 2) if has_lsl else 0.0
        rows.append(
            {
                "subject_id": subject_record.subject_id,
                "task_id": subject_record.task_id,
                "file": str(subject_record.file_path),
                "fixation_id": fixation_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(duration),
                "fix_x": float(fix_x),
                "fix_y": float(fix_y),
                "AOI": _get_aoi_name(float(fix_x), float(fix_y), aois),
                **(
                    assign_trial(subject_record.trial_df, midpoint)
                    if has_lsl and not subject_record.trial_df.empty
                    else empty_trial_info()
                ),
            },
        )
    raw = pd.DataFrame(rows)
    summary = with_subject_metadata(
        summarize_fixations(
            raw,
            subject_record.trial_df,
            overall_duration_seconds(time_series.to_numpy()),
        ),
        subject_record.subject_id,
        subject_record.task_id,
        subject_record.file_path,
    )
    summary["fixation_rate"] = summary.apply(
        lambda row: row["n_fixations"] / row["total_time"] if row["total_time"] > 0 else 0.0,
        axis=1,
    )
    for aoi in aois:
        name = str(aoi["name"])
        aoi_rows = raw.loc[raw["AOI"] == name] if not raw.empty else raw
        summary[f"{name}_fixation_count"] = int(len(aoi_rows))
        summary[f"{name}_fixation_duration"] = float(aoi_rows["duration"].sum()) if not aoi_rows.empty else 0.0
    return raw, summary


async def extract_fixations_from_config_aio(
    config_path: str | Path,
    *,
    save_output: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # This function is now a stub, as there are no subjects in the config.
    return pd.DataFrame(), pd.DataFrame()
