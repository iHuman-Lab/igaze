import warnings
from contextlib import suppress
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
    from igaze import _eyetracking_common as common
except ModuleNotFoundError:
    import _eyetracking_common as common

FIXATION_METRICS = common.FIXATION_METRICS
SACCADE_METRICS = common.SACCADE_METRICS
TRIAL_MERGE_KEYS = common.TRIAL_MERGE_KEYS
COUNT_OUTCOMES = ["saved_victims", "step_count", "n_fixations", "n_saccades"]
CONTINUOUS_OUTCOMES = [
    "mean_fixation_duration",
    "total_fixation_time",
    "fixation_rate",
    "mean_saccade_duration",
    "total_saccade_time",
    "mean_amplitude",
    "saccade_rate",
]


def _load_eye_metric_summaries(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, et_cfg = common.load_eyetracking_config(config_path)
    output_cfg = et_cfg.get("output", {})

    def _load_summary(csv_key: str, extractor) -> pd.DataFrame:
        path = output_cfg.get(csv_key)
        if not path:
            return pd.DataFrame()
        resolved = common.resolve_path(config_path.parent.parent, path)
        df = pd.read_csv(resolved) if resolved.exists() else extractor(config_path)[1]
        if "subject_id" in df.columns and "participant_id" not in df.columns:
            df = df.rename(columns={"subject_id": "participant_id"})
        if "participant_id" in df.columns:
            df["participant_id"] = df["participant_id"].astype(str)
        return df

    from igaze.fixation import extract_fixations_from_config
    from igaze.saccade import extract_saccades_from_config

    return _load_summary("summary_csv", extract_fixations_from_config), _load_summary(
        "saccades_summary_csv", extract_saccades_from_config
    )


def extract_trials_from_xdf(config_path: str | Path) -> pd.DataFrame:
    config_path = Path(config_path)
    project_root, et_cfg = common.load_eyetracking_config(config_path)
    frames = []
    for subject in et_cfg["subjects"]:
        file_path = common.resolve_path(project_root, subject["file"])
        _, game_df = common.load_subject_data(file_path)
        if game_df.empty:
            continue
        game_df = game_df.copy()
        game_df["subject_id"] = str(subject["subject_id"])
        game_df["expertise"] = subject.get("expertise", "unknown")
        frames.append(game_df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values(["subject_id", "_timestamp"])
    df = df.reset_index(drop=True)
    df["trial_id"] = df.groupby("subject_id", group_keys=False).apply(
        lambda group: (
            (group["prompt_type"] != group["prompt_type"].shift())
            | (group["llm_model"] != group["llm_model"].shift())
            | (group["llm_provider"] != group["llm_provider"].shift())
        ).cumsum()
    ).reset_index(drop=True)
    return df


def run_glmm_models_from_trials(trial_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["subject_id", "trial_id", "llm_provider", "prompt_type", "llm_model", "expertise"]
    agg = {col: "last" for col in ("saved_victims", "step_count") if col in trial_df.columns}
    return trial_df.groupby(group_cols, dropna=False).agg(agg).reset_index()


def _normalize_model_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for target, source in [("participant_id", "subject_id"), ("llm_provider", "AI"), ("prompt_type", "Prompt")]:
        if target not in df.columns and source in df.columns:
            df[target] = df[source]
    mapping = {0: "sparse", 1: "detailed", "0": "sparse", "1": "detailed"}
    if "prompt_type" not in df.columns and "Detailed" in df.columns:
        df["prompt_type"] = df["Detailed"].map(mapping)
    if "llm_provider" not in df.columns and "Gemini" in df.columns:
        df["llm_provider"] = df["Gemini"].map({0: "other", 1: "gemini", "0": "other", "1": "gemini"})
    if "expertise" not in df.columns:
        df["expertise"] = "unknown"
    df["participant_id"] = df["participant_id"].astype(str)
    for col in ("llm_provider", "prompt_type"):
        df[col] = df[col].fillna("unknown").astype(str)
    df["expertise"] = df["expertise"].fillna("unknown").astype(str).str.strip().str.lower()
    valid = {"expert": "expert", "novice": "novice", "unknown": "unknown", "": "unknown"}
    df["expertise"] = df["expertise"].map(valid).fillna(df["expertise"])
    invalid = sorted(set(df["expertise"].dropna()) - {"expert", "novice", "unknown"})
    if invalid:
        raise ValueError(f"Invalid expertise value(s). Use only 'expert' or 'novice': {invalid}")
    return df


def prepare_glmm_df(config_path: str | Path) -> pd.DataFrame:
    df = _normalize_model_df(run_glmm_models_from_trials(extract_trials_from_xdf(config_path)))
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    for metrics in _load_eye_metric_summaries(Path(config_path)):
        if not metrics.empty:
            df = df.merge(metrics, on=TRIAL_MERGE_KEYS, how="left")
    df["Gemini"] = df["llm_model"].astype(str).str.lower().str.contains("gemini").astype(int)
    df["Detailed"] = df["prompt_type"].astype(str).str.lower().str.contains("detail").astype(int)
    for col in COUNT_OUTCOMES + CONTINUOUS_OUTCOMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("saved_victims", "step_count"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


def _fixed_effects_formula(outcome: str, df: pd.DataFrame) -> str:
    return f"{outcome} ~ C(llm_provider) * C(prompt_type)" + (
        " + C(expertise)" if "expertise" in df.columns and df["expertise"].nunique(dropna=True) > 1 else ""
    )


def _fit_mixedlm(df: pd.DataFrame, outcome: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        return smf.mixedlm(_fixed_effects_formula(outcome, df), df, groups=df["participant_id"]).fit()


def _fit_poisson_glmm(df: pd.DataFrame, outcome: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        return PoissonBayesMixedGLM.from_formula(
            _fixed_effects_formula(outcome, df), {"participant": "0 + C(participant_id)"}, df
        ).fit_vb()


def _run_outcome_models(df: pd.DataFrame, outcomes: list[str], fit_fn) -> dict[str, object]:
    results = {}
    for outcome in outcomes:
        if outcome in df.columns and len(df.dropna(subset=[outcome])):
            with suppress(Exception):
                results[outcome] = fit_fn(df.dropna(subset=[outcome]).copy(), outcome)
    return results


def run_glmm_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _normalize_model_df(df)
    for col in COUNT_OUTCOMES + CONTINUOUS_OUTCOMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["participant_id", "llm_provider", "prompt_type"]).copy()
    mixed = _run_outcome_models(df, CONTINUOUS_OUTCOMES, _fit_mixedlm)
    count = _run_outcome_models(df, COUNT_OUTCOMES, _fit_poisson_glmm)
    mixed_rows = [
        {"outcome": outcome, "term": term, "coef": model.params[term], "se": model.bse.get(term)}
        for outcome, model in mixed.items()
        for term in model.params.index
    ]
    count_rows = [
        {"outcome": outcome, "term": term, "coef": coef}
        for outcome, model in count.items()
        for term, coef in zip(model.model.exog_names, model.fe_mean)
    ]
    return pd.DataFrame(mixed_rows), pd.DataFrame(count_rows)
