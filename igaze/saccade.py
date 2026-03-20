from __future__ import annotations

from pathlib import Path

import pandas as pd

from igaze.detectors import saccade_detection

try:
    from igaze import _eyetracking_common as common
except ModuleNotFoundError:
    import _eyetracking_common as common



def extract_saccades_from_config(config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run saccade detection for all subjects in config."""
    config_path = Path(config_path)
    project_root, et_cfg = common.load_eyetracking_config(config_path)
    x_col = et_cfg["columns"]["x"]
    y_col = et_cfg["columns"]["y"]
    time_col = et_cfg["columns"]["time"]
    minlen = et_cfg.get("minlen", 5)
    maxvel = et_cfg.get("maxvel", 40)
    maxacc = et_cfg.get("maxacc", 340)
    missing = et_cfg.get("missing", 0.0)

    raw_saccades, summaries = [], []

    for subject in et_cfg["subjects"]:
        subject_record = common.load_subject_record(project_root, subject)
        missing_cols = {x_col, y_col, time_col} - set(subject_record.eye_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in {subject_record.file_path}: {sorted(missing_cols)}")

        x_series = pd.to_numeric(subject_record.eye_df[x_col], errors="coerce")
        y_series = pd.to_numeric(subject_record.eye_df[y_col], errors="coerce")
        time_series = pd.to_numeric(subject_record.eye_df[time_col], errors="coerce")
        x_series, y_series = common.scale_coordinates(x_series, y_series, et_cfg)

        _, end_saccades = saccade_detection(
            x_series.to_numpy(),
            y_series.to_numpy(),
            time_series.to_numpy(),
            missing=missing,
            minlen=minlen,
            maxvel=maxvel,
            maxacc=maxacc,
        )

        has_lsl = "_lsl_timestamp" in subject_record.eye_df.columns
        lsl_series = subject_record.eye_df["_lsl_timestamp"] if has_lsl else pd.Series(dtype=float)
        subject_saccades = []

        for saccade_id, saccade_end in enumerate(end_saccades, start=1):
            start_time, end_time, duration, x_start, y_start, x_end, y_end = saccade_end
            start_idx = common.nearest_index(time_series, start_time)
            end_idx = common.nearest_index(time_series, end_time)
            amplitude = ((x_end - x_start) ** 2 + (y_end - y_start) ** 2) ** 0.5

            trial_info = common.empty_trial_info()
            if has_lsl and not subject_record.trial_df.empty:
                midpoint = float((lsl_series.iloc[start_idx] + lsl_series.iloc[end_idx]) / 2)
                trial_info = common.assign_trial(subject_record.trial_df, midpoint)

            row = {
                "subject_id": subject_record.subject_id,
                "task_id": subject_record.task_id,
                "file": str(subject_record.file_path),
                "saccade_id": saccade_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(duration),
                "x_start": float(x_start),
                "y_start": float(y_start),
                "x_end": float(x_end),
                "y_end": float(y_end),
                "amplitude": float(amplitude),
                **trial_info,
            }
            raw_saccades.append(row)
            subject_saccades.append(row)

        summary_group = common.summarize_saccades(
            pd.DataFrame(subject_saccades),
            subject_record.trial_df,
            common.overall_duration_seconds(time_series.to_numpy()),
        )

        for _, summary_row in summary_group.iterrows():
            total_time = float(summary_row["total_time"])
            n_saccades = int(summary_row["n_saccades"])
            saccade_rate = n_saccades / total_time if total_time > 0 else 0.0
            summaries.append(
                {
                    "subject_id": subject_record.subject_id,
                    "task_id": subject_record.task_id,
                    "file": str(subject_record.file_path),
                    "trial_id": summary_row["trial_id"],
                    "n_saccades": n_saccades,
                    "mean_saccade_duration": float(summary_row["mean_saccade_duration"]),
                    "total_saccade_time": float(summary_row["total_saccade_time"]),
                    "mean_amplitude": float(summary_row["mean_amplitude"]),
                    "total_time": total_time,
                    "saccade_rate": float(saccade_rate),
                },
            )

    return pd.DataFrame(raw_saccades), pd.DataFrame(summaries)
