import asyncio
from pathlib import Path

import yaml

from igaze.fixation import export_eyetracker_timeline_from_config, extract_fixations_from_config
from igaze.fixation_aio import extract_fixations_from_config_aio
from igaze.glmm import prepare_glmm_df, run_glmm_models
from igaze.saccade import extract_saccades_from_config
from igaze.xdf import convert_subjects_from_config as xdf_config
from utils import skip_run

try:
    from igaze.backward_saccade import extract_backward_saccades_from_config
except ModuleNotFoundError:
    extract_backward_saccades_from_config = None

# The configuration file
config_path = "configs/config.yml"
project_root = Path(__file__).resolve().parent

with Path(config_path).open("r", encoding="utf-8") as file_handle:
    config = yaml.safe_load(file_handle)

with skip_run("run", "xdf to csv") as check, check():
    csv_outputs = xdf_config(config_path)

with skip_run("run", "fixation_extraction") as check, check():
    raw_fixations, fixation_summary = extract_fixations_from_config(config_path, save_output=True)

with skip_run("run", "eyetracker_timeline") as check, check():
    eyetracker_timeline, llm_periods = export_eyetracker_timeline_from_config(config_path, save_output=True)

with skip_run("run", "fixation_aio_extraction") as check, check():
    # Run the async AOI-aware fixation extraction
    raw_fixations_aio, fixation_aio_summary = asyncio.run(extract_fixations_from_config_aio(config_path))

with skip_run("run", "saccade_extraction") as check, check():
    raw_saccades, saccade_summary = extract_saccades_from_config(config_path)

with skip_run("run", "backward_saccade_extraction") as check, check():
    if extract_backward_saccades_from_config is None:
        raw_backward_saccades = None
        backward_saccade_summary = None
    else:
        raw_backward_saccades, backward_saccade_summary = extract_backward_saccades_from_config(config_path)

with skip_run("run", "glmm") as check, check():
    df = prepare_glmm_df(config_path)
    glmm_output_dir = (project_root / "outputs" / "glmm").resolve()
    glmm_output_dir.mkdir(parents=True, exist_ok=True)
    prepared_glmm_out = glmm_output_dir / "prepared_glmm_data.csv"
    continuous_glmm_out = glmm_output_dir / "glmm_continuous_results.csv"
    count_glmm_out = glmm_output_dir / "glmm_count_results.csv"

    df.to_csv(prepared_glmm_out, index=False)
    continuous_results, count_results = run_glmm_models(df)
    continuous_results.to_csv(continuous_glmm_out, index=False)
    count_results.to_csv(count_glmm_out, index=False)
