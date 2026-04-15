import asyncio

from igaze.fixation import extract_fixations_from_config
from igaze.fixation_aio import extract_fixations_from_config_aio
from igaze.saccade import extract_saccades_from_config
from utils import skip_run

# The configuration file
config_path = "configs/config.yml"

with skip_run("run", "fixation_extraction") as check, check():
    raw_fixations, fixation_summary = extract_fixations_from_config(config_path)

with skip_run("run", "fixation_aio_extraction") as check, check():
    raw_fixations_aio, fixation_aio_summary = asyncio.run(extract_fixations_from_config_aio(config_path))

with skip_run("run", "saccade_extraction") as check, check():
    raw_saccades, saccade_summary = extract_saccades_from_config(config_path)
