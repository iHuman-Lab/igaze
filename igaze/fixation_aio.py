"""AOI-aware fixation extraction for config-driven eye-tracking datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from igaze.detectors import fixation_detection

try:
    from igaze import _eyetracking_common as common
except ModuleNotFoundError:
    import _eyetracking_common as common


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _get_aoi_name(fix_x: float, fix_y: float, aois: list[dict]) -> str | None:
    for aoi in aois:
        if aoi["x_min"] <= fix_x <= aoi["x_max"] and aoi["y_min"] <= fix_y <= aoi["y_max"]:
            return str(aoi["name"])
    return None


def _subject_fixation_aoi_frames(
    subject_record: common.SubjectRecord,
    et_cfg: dict,
    aois: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = et_cfg["columns"]
    time_series = pd.to_numeric(subject_record.eye_df[columns["time"]], errors="coerce")
    x_series, y_series = common.scale_coordinates(
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
        start_idx = common.nearest_index(time_series, start_time)
        end_idx = common.nearest_index(time_series, end_time)
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
                    common.assign_trial(subject_record.trial_df, midpoint)
                    if has_lsl and not subject_record.trial_df.empty
                    else common.empty_trial_info()
                ),
            },
        )
    raw = pd.DataFrame(rows)
    summary = common.with_subject_metadata(
        common.summarize_fixations(
            raw,
            subject_record.trial_df,
            common.overall_duration_seconds(time_series.to_numpy()),
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


async def extract_fixations_from_config_aio(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config_path = Path(config_path)
    project_root, et_cfg = common.load_eyetracking_config(config_path)
    aois = common.scale_aoi_bounds(et_cfg.get("AOIs", []), et_cfg)
    all_raw, all_summaries = [], []
    for subject in et_cfg.get("subjects", []):
        subject_record = common.load_subject_record(project_root, subject)
        raw, summary = _subject_fixation_aoi_frames(subject_record, et_cfg, aois)
        all_raw.append(raw)
        all_summaries.append(summary)
    return _concat_frames(all_raw), _concat_frames(all_summaries)
