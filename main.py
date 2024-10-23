"""Main file."""

from pathlib import Path

import pandas as pd
import yaml

from igaze.detectors import fixation_detection
from utils import skip_run

# The configuration file
config_path = "configs/config.yml"
with Path("configs/config.yml").open() as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

with skip_run("run", "test_data_loading") as check, check():
    data = pd.read_csv(config["data_path"])

    start_fixation, end_fixation = fixation_detection(
        data["avg_x"],
        data["avg_y"],
        time=data["time"],
        mindur=10,
        maxdist=0.5,
    )

    print(start_fixation, end_fixation)
