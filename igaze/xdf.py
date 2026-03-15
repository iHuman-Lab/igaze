from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path

import pandas as pd
import pyxdf
import yaml


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "stream"


def _resolve_path(project_root: Path, file_name: str) -> Path:
    file_path = Path(file_name)
    if file_path.is_absolute():
        return file_path
    return (project_root / file_path).resolve()


def _channel_labels(stream_info: dict, sample_width: int) -> list[str]:
    try:
        channels = stream_info["desc"][0]["channels"][0]["channel"]
        labels = [channel.get("label", [f"ch{index}"])[0] for index, channel in enumerate(channels)]
        if len(labels) >= sample_width:
            return labels[:sample_width]
    except (KeyError, IndexError, TypeError):
        pass
    return [f"ch{index}" for index in range(sample_width)]


def _stream_dataframe(stream: dict) -> pd.DataFrame:
    time_series = stream.get("time_series", [])
    time_stamps = stream.get("time_stamps", [])

    if len(time_series) == 0:
        df = pd.DataFrame()
    else:
        first_row = time_series[0]
        sample_width = len(first_row) if hasattr(first_row, "__len__") and not isinstance(first_row, str) else 1

        if sample_width == 1:
            rows = []
            for row in time_series:
                if isinstance(row, str):
                    rows.append([row])
                else:
                    rows.append([row[0]])
        else:
            rows = time_series

        labels = _channel_labels(stream.get("info", {}), sample_width)
        df = pd.DataFrame(rows, columns=labels)

    if len(time_stamps) == len(df):
        df.insert(0, "lsl_timestamp", time_stamps)

    return df


def _extract_game_rows(stream: dict) -> list[dict]:
    time_series = stream.get("time_series", [])
    time_stamps = stream.get("time_stamps", [])
    game_rows: list[dict] = []

    for ts, row in zip(time_stamps, time_series):
        try:
            payload = json.loads(row[0])
        except (TypeError, IndexError, json.JSONDecodeError):
            continue
        if "prompt_type" not in payload or "llm_model" not in payload:
            continue
        payload["_timestamp"] = float(ts)
        payload["llm_provider"] = payload.get("llm_provider", payload.get("provider"))
        game_rows.append(payload)

    return game_rows


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


def _annotate_gaze_dataframe(gaze_df: pd.DataFrame, trial_df: pd.DataFrame) -> pd.DataFrame:
    annotated = gaze_df.copy()
    annotated["trial_id"] = None
    annotated["llm_model"] = None
    annotated["prompt_type"] = None
    annotated["llm_provider"] = None
    annotated["trial_start"] = None
    annotated["trial_end"] = None

    if trial_df.empty or "lsl_timestamp" not in annotated.columns:
        return annotated

    for _, trial_row in trial_df.iterrows():
        mask = (
            (annotated["lsl_timestamp"] >= trial_row["trial_start"])
            & (annotated["lsl_timestamp"] <= trial_row["trial_end"])
        )
        annotated.loc[mask, "trial_id"] = int(trial_row["trial_id"])
        annotated.loc[mask, "llm_model"] = trial_row["llm_model"]
        annotated.loc[mask, "prompt_type"] = trial_row["prompt_type"]
        annotated.loc[mask, "llm_provider"] = trial_row["llm_provider"]
        annotated.loc[mask, "trial_start"] = float(trial_row["trial_start"])
        annotated.loc[mask, "trial_end"] = float(trial_row["trial_end"])

    return annotated


def convert_xdf_to_csv(xdf_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    xdf_path = Path(xdf_path).resolve()
    target_dir = Path(output_dir).resolve() if output_dir else xdf_path.with_suffix("")
    target_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    written_files: list[Path] = []
    game_rows: list[dict] = []

    for stream in streams:
        stream_type = stream.get("info", {}).get("type", [""])[0]
        stream_name = stream.get("info", {}).get("name", [""])[0]
        if stream_type == "Gaze":
            continue
        if stream_name == "SARGame" or stream_type == "GameState":
            game_rows.extend(_extract_game_rows(stream))

    trial_df = _build_trial_windows(pd.DataFrame(game_rows))

    for index, stream in enumerate(streams):
        info = stream.get("info", {})
        stream_name = info.get("name", [f"stream_{index}"])[0]
        stream_type = info.get("type", ["unknown"])[0]
        safe_name = _safe_name(stream_name)
        safe_type = _safe_name(stream_type)
        csv_path = target_dir / f"{index:02d}_{safe_name}_{safe_type}.csv"

        df = _stream_dataframe(stream)
        if stream_type == "Gaze":
            df = _annotate_gaze_dataframe(df, trial_df)
        df.to_csv(csv_path, index=False)
        written_files.append(csv_path)

    return written_files


def convert_subjects_from_config(
    config_path: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, list[Path]]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent.resolve()

    with config_path.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    et_cfg = config["eyetracking"]
    target_root = Path(output_root).resolve() if output_root else (project_root / "outputs" / "xdf_csv")
    target_root.mkdir(parents=True, exist_ok=True)

    written_by_subject: dict[str, list[Path]] = {}
    for subject in et_cfg["subjects"]:
        subject_id = str(subject["subject_id"])
        xdf_path = _resolve_path(project_root, subject["file"])
        subject_dir = target_root / f"subject_{subject_id}"
        written_by_subject[subject_id] = convert_xdf_to_csv(xdf_path, subject_dir)

    return written_by_subject