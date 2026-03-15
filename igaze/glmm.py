import warnings
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


def _load_eye_metric_summaries(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, et_cfg = common.load_eyetracking_config(config_path)
    output_cfg = et_cfg.get("output", {})

    def _load_summary(csv_key: str, extractor):
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

    return (
        _load_summary("summary_csv", extract_fixations_from_config),
        _load_summary("saccades_summary_csv", extract_saccades_from_config),
    )


def extract_trials_from_xdf(config_path):
    """Load all participant XDF files and return a trial-level DataFrame."""
    config_path = Path(config_path)
    project_root, et_cfg = common.load_eyetracking_config(config_path)
    all_frames = []

    for subj in et_cfg["subjects"]:
        subject_id = str(subj["subject_id"])
        file_path = common.resolve_path(project_root, subj["file"])
        _, game_df = common.load_subject_data(file_path)
        if game_df.empty:
            continue
        game_df = game_df.copy()
        game_df["subject_id"] = subject_id
        game_df["expertise"] = subj.get("expertise")
        all_frames.append(game_df)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values(["subject_id", "_timestamp"]).reset_index(drop=True)
    df["trial_id"] = (
        df.groupby("subject_id", group_keys=False)
        .apply(lambda g: (
            (g["prompt_type"] != g["prompt_type"].shift(1))
            | (g["llm_model"] != g["llm_model"].shift(1))
            | (g["llm_provider"] != g["llm_provider"].shift(1))
        ).cumsum())
        .reset_index(drop=True)
    )
    return df


def run_glmm_models_from_trials(trial_df):
    """Aggregate to end-of-mission values per subject per condition."""
    group_cols = [
        "subject_id",
        "trial_id",
        "llm_provider",
        "prompt_type",
        "llm_model",
        "expertise",
    ]
    return (
        trial_df.groupby(group_cols, dropna=False)
        .agg(
            saved_victims=("saved_victims", "last"),
            step_count=("step_count", "last"),
        )
        .reset_index()
    )


def prepare_glmm_df(config_path):
    """Build a GLMM-ready DataFrame from XDF files.

    Keeps categorical predictors for mixed-effects formulas:
      - llm_provider
      - prompt_type
      - expertise
      - participant_id
    Outcomes: saved_victims (count), step_count (count).
    """
    trial_df = extract_trials_from_xdf(config_path)
    agg_df = run_glmm_models_from_trials(trial_df)
    agg_df = agg_df.rename(columns={"subject_id": "participant_id"})
    for col in ("llm_provider", "prompt_type", "expertise"):
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].fillna("unknown").astype(str)

    agg_df["participant_id"] = agg_df["participant_id"].astype(str)
    agg_df["trial_id"] = pd.to_numeric(agg_df["trial_id"], errors="coerce")

    fixation_metrics, saccade_metrics = _load_eye_metric_summaries(Path(config_path))

    if not fixation_metrics.empty:
        agg_df = agg_df.merge(fixation_metrics, on=TRIAL_MERGE_KEYS, how="left")
    if not saccade_metrics.empty:
        agg_df = agg_df.merge(saccade_metrics, on=TRIAL_MERGE_KEYS, how="left")

    agg_df["Gemini"] = (agg_df["llm_model"].str.lower().str.contains("gemini")).astype(int)
    agg_df["Detailed"] = (agg_df["prompt_type"].str.lower().str.contains("detail")).astype(int)
    agg_df["saved_victims"] = pd.to_numeric(agg_df["saved_victims"], errors="coerce").fillna(0).astype(int)
    agg_df["step_count"] = pd.to_numeric(agg_df["step_count"], errors="coerce").fillna(0).astype(int)

    for column in FIXATION_METRICS + SACCADE_METRICS:
        if column in agg_df.columns:
            agg_df[column] = pd.to_numeric(agg_df[column], errors="coerce")

    return agg_df


def _normalize_model_df(df: pd.DataFrame) -> pd.DataFrame:
    n = df.copy()

    for target, source in [("participant_id", "subject_id"), ("llm_provider", "AI"), ("prompt_type", "Prompt")]:
        if target not in n.columns and source in n.columns:
            n[target] = n[source]

    if "prompt_type" not in n.columns and "Detailed" in n.columns:
        n["prompt_type"] = n["Detailed"].map({0: "sparse", 1: "detailed", "0": "sparse", "1": "detailed"})
    if "llm_provider" not in n.columns and "Gemini" in n.columns:
        n["llm_provider"] = n["Gemini"].map({0: "other", 1: "gemini", "0": "other", "1": "gemini"})
    if "expertise" not in n.columns:
        n["expertise"] = "unknown"

    n["participant_id"] = n["participant_id"].astype(str)
    for col in ("llm_provider", "prompt_type"):
        n[col] = n[col].fillna("unknown").astype(str)
    n["expertise"] = n["expertise"].fillna("unknown").astype(str).str.strip().str.lower()
    n["expertise"] = n["expertise"].map(
        {"expert": "expert", "novice": "novice", "unknown": "unknown", "": "unknown"},
    ).fillna(n["expertise"])

    invalid = sorted(set(n["expertise"].dropna()) - {"expert", "novice", "unknown"})
    if invalid:
        raise ValueError(f"Invalid expertise value(s). Use only 'expert' or 'novice': {invalid}")

    return n


def _fixed_effects_formula(outcome: str, df: pd.DataFrame) -> str:
    formula = f"{outcome} ~ C(llm_provider) * C(prompt_type)"
    if "expertise" in df.columns and df["expertise"].nunique(dropna=True) > 1:
        formula += " + C(expertise)"
    return formula


def _fit_mixedlm(sub: pd.DataFrame, outcome: str):
    formula = _fixed_effects_formula(outcome, sub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        return smf.mixedlm(
            formula=formula,
            data=sub,
            groups=sub["participant_id"],
        ).fit()


def _fit_poisson_glmm(sub: pd.DataFrame, outcome: str):
    formula = _fixed_effects_formula(outcome, sub)
    vcf = {"participant": "0 + C(participant_id)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        return PoissonBayesMixedGLM.from_formula(formula, vcf, sub).fit_vb()


def _collect_model_rows(outcome: str, model, mixedlm_rows: list[dict], poisson_rows: list[dict]) -> None:
    if hasattr(model, "params") and hasattr(model, "bse"):
        for term in model.params.index:
            mixedlm_rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coef": model.params[term],
                    "se": model.bse[term] if term in model.bse.index else None,
                },
            )

    if hasattr(model, "fe_mean"):
        exog_names = model.model.exog_names
        for term, coef in zip(exog_names, model.fe_mean):
            poisson_rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coef": coef,
                },
            )


def _run_outcome_models(
    df: pd.DataFrame, outcomes: list[str], fit_fn, results: dict,
) -> None:
    for outcome in outcomes:
        if outcome not in df.columns or not len(df.dropna(subset=[outcome])):
            continue
        try:
            results[outcome] = fit_fn(df.dropna(subset=[outcome]).copy(), outcome)
        except Exception:
            pass


def run_glmm_models(df):
    df = _normalize_model_df(df)

    count_outcomes = ["saved_victims", "step_count", "n_fixations", "n_saccades"]
    continuous_outcomes = [
        "mean_fixation_duration",
        "total_fixation_time",
        "fixation_rate",
        "mean_saccade_duration",
        "total_saccade_time",
        "mean_amplitude",
        "saccade_rate",
    ]

    base_cols = ["participant_id", "llm_provider", "prompt_type"]
    for col in continuous_outcomes + count_outcomes:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=base_cols).copy()
    results = {}

    _run_outcome_models(df, continuous_outcomes, _fit_mixedlm, results)
    _run_outcome_models(df, count_outcomes, _fit_poisson_glmm, results)

    mixedlm_rows = []
    poisson_rows = []

    for outcome, model in results.items():
        _collect_model_rows(outcome, model, mixedlm_rows, poisson_rows)

    mixedlm_summary_df = pd.DataFrame(mixedlm_rows)
    poisson_summary_df = pd.DataFrame(poisson_rows)

    return mixedlm_summary_df, poisson_summary_df
