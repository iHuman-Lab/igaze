"""Shared loading and trial-label utilities for eye-tracking analyses."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyxdf
import yaml


FIXATION_METRICS = [
    "n_fixations",
    "mean_fixation_duration",
    "total_fixation_time",
    "fixation_rate",
]

SACCADE_METRICS = [
    "n_saccades",
    "mean_saccade_duration",
    "total_saccade_time",
    "mean_amplitude",
    "saccade_rate",
]

TRIAL_MERGE_KEYS = [
    "participant_id",
    "trial_id",
    "llm_model",
    "prompt_type",
    "llm_provider",
]

TRIAL_INFO_FIELDS = (
    "trial_id",
    "llm_model",
    "prompt_type",
    "llm_provider",
)

DEFAULT_EYETRACKING_OUTPUTS = {
    "raw_fixations_csv": "outputs/eyetracking/fixations_long.csv",
    "summary_csv": "outputs/eyetracking/fixations_summary.csv",
    "raw_fixations_aio_csv": "outputs/eyetracking/fixations_long_aio.csv",
    "summary_aio_csv": "outputs/eyetracking/fixations_summary_aio.csv",
    "eyetracker_timeline_csv": "outputs/eyetracking/eyetracker_timeline.csv",
    "llm_periods_csv": "outputs/eyetracking/llm_periods.csv",
    "raw_saccades_csv": "outputs/eyetracking/saccades_long.csv",
    "saccades_summary_csv": "outputs/eyetracking/saccades_summary.csv",
}

DEFAULT_EYETRACKING_CONFIG = Path("configs") / "config.yml"


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    task_id: str | None
    file_path: Path
    eye_df: pd.DataFrame
    game_df: pd.DataFrame
    trial_df: pd.DataFrame


def _resolve_path(project_root: Path, file_name: str) -> Path:
    file_path = Path(file_name)
    return file_path if file_path.is_absolute() else (project_root / file_path).resolve()


def _load_eyetracking_config(config_path: Path) -> tuple[Path, dict]:
    with config_path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    return config_path.parent.parent.resolve(), config["eyetracking"]


def _default_eyetracking_config_path(module_file: str | Path) -> Path:
    return Path(module_file).resolve().parent.parent / DEFAULT_EYETRACKING_CONFIG


def _get_output_config(et_cfg: dict) -> dict:
    return {**DEFAULT_EYETRACKING_OUTPUTS, **et_cfg.get("output", {})}


def _get_output_path(project_root: Path, et_cfg: dict, key: str) -> Path:
    return _resolve_path(project_root, _get_output_config(et_cfg)[key])


def _empty_trial_info() -> dict:
    return dict.fromkeys(TRIAL_INFO_FIELDS)


def _extract_channel_labels(stream_info: dict, default_count: int) -> list[str]:
    try:
        channels = stream_info["desc"][0]["channels"][0]["channel"]
        return [channel["label"][0] for channel in channels]
    except (KeyError, IndexError, TypeError):
        return [f"ch{index}" for index in range(default_count)]


def _find_game_csv(file_path: Path) -> Path | None:
    if file_path.suffix.lower() != ".csv":
        return None

    candidates = sorted(file_path.parent.glob("*_SARGame_*.csv"))
    if not candidates:
        candidates = sorted(file_path.parent.glob("*SARGame*.csv"))

    for candidate in candidates:
        if candidate.resolve() != file_path.resolve():
            return candidate
    return None


def _load_csv_game_data(file_path: Path) -> pd.DataFrame:
    game_csv = pd.read_csv(file_path)
    if game_csv.empty or "lsl_timestamp" not in game_csv.columns:
        return pd.DataFrame()

    payload_columns = [column for column in game_csv.columns if column != "lsl_timestamp"]
    if not payload_columns:
        return pd.DataFrame()

    payload_column = payload_columns[0]
    game_rows = []
    for _, row in game_csv.iterrows():
        try:
            payload = json.loads(row[payload_column])
        except (TypeError, json.JSONDecodeError):
            continue
        if "prompt_type" not in payload or "llm_model" not in payload:
            continue
        payload["_timestamp"] = float(row["lsl_timestamp"])
        payload["llm_provider"] = payload.get("llm_provider", payload.get("provider"))
        game_rows.append(payload)

    return pd.DataFrame(game_rows)


def _load_csv_eye_data(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    eye_df = pd.read_csv(file_path)
    if "lsl_timestamp" in eye_df.columns and "_lsl_timestamp" not in eye_df.columns:
        eye_df["_lsl_timestamp"] = pd.to_numeric(eye_df["lsl_timestamp"], errors="coerce")

    game_csv_path = _find_game_csv(file_path)
    game_df = _load_csv_game_data(game_csv_path) if game_csv_path else pd.DataFrame()
    return eye_df, game_df


def _load_xdf_data(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        streams, _ = pyxdf.load_xdf(str(file_path))
    gaze_df = pd.DataFrame()
    game_rows = []

    for stream in streams:
        stream_type = stream["info"].get("type", [""])[0]
        time_series = stream.get("time_series", [])
        time_stamps = stream.get("time_stamps", [])

        if stream_type == "Gaze":
            default_count = len(time_series[0]) if len(time_series) > 0 else 3
            labels = _extract_channel_labels(stream["info"], default_count)
            gaze_df = pd.DataFrame(time_series, columns=labels)
            gaze_df["_lsl_timestamp"] = time_stamps
            continue

        for ts, row in zip(time_stamps, time_series):
            try:
                payload = json.loads(row[0])
            except (TypeError, IndexError, json.JSONDecodeError):
                continue
            if "prompt_type" not in payload or "llm_model" not in payload:
                continue
            payload["_timestamp"] = ts
            payload["llm_provider"] = payload.get("llm_provider", payload.get("provider"))
            game_rows.append(payload)

    if gaze_df.empty:
        raise ValueError(f"No Gaze stream found in {file_path}")

    return gaze_df, pd.DataFrame(game_rows)


def _load_subject_data(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_xdf_data(file_path) if file_path.suffix.lower() == ".xdf" else _load_csv_eye_data(file_path)


def _load_subject_record(project_root: Path, subject: dict) -> SubjectRecord:
    file_path = _resolve_path(project_root, subject["file"])
    eye_df, game_df = _load_subject_data(file_path)
    return SubjectRecord(
        subject_id=str(subject["subject_id"]),
        task_id=subject.get("task_id"),
        file_path=file_path,
        eye_df=eye_df,
        game_df=game_df,
        trial_df=_build_trial_windows(game_df),
    )


def _scale_coordinates(x: pd.Series, y: pd.Series, et_cfg: dict) -> tuple[pd.Series, pd.Series]:
    if str(et_cfg.get("coordinate_system", "pixels")).lower() != "normalized":
        return x, y

    display_cfg = et_cfg.get("display", {})
    return x * float(display_cfg.get("width_px", 1920)), y * float(display_cfg.get("height_px", 1080))


def _scale_aoi_bounds(aois: list[dict], et_cfg: dict) -> list[dict]:
    if str(et_cfg.get("coordinate_system", "pixels")).lower() != "normalized":
        return aois

    display_cfg = et_cfg.get("display", {})
    width_px = float(display_cfg.get("width_px", 1920))
    height_px = float(display_cfg.get("height_px", 1080))
    return [
        {
            **aoi,
            "x_min": float(aoi["x_min"]) * width_px,
            "x_max": float(aoi["x_max"]) * width_px,
            "y_min": float(aoi["y_min"]) * height_px,
            "y_max": float(aoi["y_max"]) * height_px,
        }
        for aoi in aois
    ]


def _nearest_index(series: pd.Series, value: float) -> int:
    return int((series - value).abs().idxmin())


def _build_trial_windows(game_df: pd.DataFrame) -> pd.DataFrame:
    if game_df.empty:
        return pd.DataFrame()

    game_df = game_df.sort_values("_timestamp").reset_index(drop=True)
    provider = game_df["llm_provider"] if "llm_provider" in game_df.columns else pd.Series([None] * len(game_df))
    trial_change = (
        (game_df["llm_model"] != game_df["llm_model"].shift(1))
        | (game_df["prompt_type"] != game_df["prompt_type"].shift(1))
        | (provider != provider.shift(1))
    )
    game_df["trial_id"] = trial_change.cumsum()

    return (
        game_df.groupby("trial_id", dropna=False)
        .agg(
            trial_start=("_timestamp", "min"),
            trial_end=("_timestamp", "max"),
            llm_model=("llm_model", "first"),
            prompt_type=("prompt_type", "first"),
            llm_provider=("llm_provider", "first"),
        )
        .reset_index()
    )


def _trial_durations(trial_df: pd.DataFrame) -> pd.DataFrame:
    if trial_df.empty:
        return pd.DataFrame()

    durations = trial_df[["trial_id", "llm_model", "prompt_type", "llm_provider"]].copy()
    durations["total_time"] = (trial_df["trial_end"] - trial_df["trial_start"]).astype(float)
    return durations


def _empty_fixation_summary(trial_df: pd.DataFrame, total_time_seconds: float) -> pd.DataFrame:
    if trial_df.empty:
        return pd.DataFrame(
            [{
                "trial_id": None,
                "llm_model": None,
                "prompt_type": None,
                "llm_provider": None,
                "n_fixations": 0,
                "mean_fixation_duration": 0.0,
                "total_fixation_time": 0.0,
                "total_time": total_time_seconds,
            }],
        )

    summary_group = _trial_durations(trial_df)
    summary_group["n_fixations"] = 0
    summary_group["mean_fixation_duration"] = 0.0
    summary_group["total_fixation_time"] = 0.0
    return summary_group


def _summarize_fixations(
    subject_fixations_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    total_time_seconds: float,
) -> pd.DataFrame:
    if subject_fixations_df.empty:
        return _empty_fixation_summary(trial_df, total_time_seconds)

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
    trial_durations = _trial_durations(trial_df)
    if trial_durations.empty:
        summary_group["total_time"] = total_time_seconds
        return summary_group

    summary_group = summary_group.merge(trial_durations, on=group_cols, how="left")
    summary_group["total_time"] = summary_group["total_time"].fillna(total_time_seconds)
    return summary_group


def _empty_saccade_summary(trial_df: pd.DataFrame, total_time: float) -> pd.DataFrame:
    if trial_df.empty:
        return pd.DataFrame(
            [{
                "trial_id": None,
                "llm_model": None,
                "prompt_type": None,
                "llm_provider": None,
                "n_saccades": 0,
                "mean_saccade_duration": 0.0,
                "total_saccade_time": 0.0,
                "mean_amplitude": 0.0,
                "total_time": total_time,
            }],
        )

    summary_group = _trial_durations(trial_df)
    summary_group["n_saccades"] = 0
    summary_group["mean_saccade_duration"] = 0.0
    summary_group["total_saccade_time"] = 0.0
    summary_group["mean_amplitude"] = 0.0
    return summary_group


def _summarize_saccades(
    subject_saccades_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    total_time_seconds: float,
) -> pd.DataFrame:
    if subject_saccades_df.empty:
        return _empty_saccade_summary(trial_df, total_time_seconds)

    group_cols = ["trial_id", "llm_model", "prompt_type", "llm_provider"]
    summary_group = (
        subject_saccades_df.groupby(group_cols, dropna=False)
        .agg(
            n_saccades=("saccade_id", "count"),
            mean_saccade_duration=("duration", "mean"),
            total_saccade_time=("duration", "sum"),
            mean_amplitude=("amplitude", "mean"),
        )
        .reset_index()
    )
    trial_durations = _trial_durations(trial_df)
    if trial_durations.empty:
        summary_group["total_time"] = total_time_seconds
        return summary_group

    summary_group = summary_group.merge(trial_durations, on=group_cols, how="left")
    summary_group["total_time"] = summary_group["total_time"].fillna(total_time_seconds)
    return summary_group


def _with_subject_metadata(df: pd.DataFrame, subject_id: str, task_id, file_path: Path) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "file", str(file_path))
    df.insert(0, "task_id", task_id)
    df.insert(0, "subject_id", subject_id)
    return df


def _overall_duration_seconds(time) -> float:
    return float(time[-1] - time[0]) / 1000.0 if len(time) > 1 else 0.0


def _assign_trial(trial_df: pd.DataFrame, midpoint_timestamp: float) -> dict:
    if trial_df.empty:
        return _empty_trial_info()

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


def _annotate_eye_samples(eye_df: pd.DataFrame, trial_df: pd.DataFrame) -> pd.DataFrame:
    annotated = eye_df.copy()
    for column in ["trial_id", "llm_model", "prompt_type", "llm_provider", "trial_start", "trial_end"]:
        annotated[column] = None

    if trial_df.empty or "_lsl_timestamp" not in annotated.columns:
        return annotated

    for _, trial_row in trial_df.iterrows():
        mask = (
            (annotated["_lsl_timestamp"] >= trial_row["trial_start"])
            & (annotated["_lsl_timestamp"] <= trial_row["trial_end"])
        )
        annotated.loc[mask, "trial_id"] = int(trial_row["trial_id"])
        annotated.loc[mask, "llm_model"] = trial_row["llm_model"]
        annotated.loc[mask, "prompt_type"] = trial_row["prompt_type"]
        annotated.loc[mask, "llm_provider"] = trial_row["llm_provider"]
        annotated.loc[mask, "trial_start"] = float(trial_row["trial_start"])
        annotated.loc[mask, "trial_end"] = float(trial_row["trial_end"])

    return annotated


load_eyetracking_config = _load_eyetracking_config
default_eyetracking_config_path = _default_eyetracking_config_path
get_output_config = _get_output_config
get_output_path = _get_output_path
SubjectRecord = SubjectRecord
resolve_path = _resolve_path
load_subject_data = _load_subject_data
load_subject_record = _load_subject_record
scale_coordinates = _scale_coordinates
scale_aoi_bounds = _scale_aoi_bounds
nearest_index = _nearest_index
build_trial_windows = _build_trial_windows
summarize_fixations = _summarize_fixations
overall_duration_seconds = _overall_duration_seconds
assign_trial = _assign_trial
empty_trial_info = _empty_trial_info
annotate_eye_samples = _annotate_eye_samples
with_subject_metadata = _with_subject_metadata
summarize_saccades = _summarize_saccades