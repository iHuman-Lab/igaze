"""Main file."""
# ruff: noqa: T201

from pathlib import Path

import pandas as pd
import yaml

from igaze.detectors import detect_blinks, find_fixations
from utils import skip_run

# The configuration file
config_path = "configs/config.yml"
with Path("configs/config.yml").open() as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

with skip_run("run", "test_features") as check, check():
    data = pd.read_csv(config["data_path"])

    fixations = find_fixations(
        data["avg_x"],
        data["avg_y"],
        time=data["time"].array,
        mindur=10,
        maxdist=0.5,
    )

    print(fixations)

    blinks = detect_blinks(
        data["avg_x"],
        data["avg_y"],
        time=data["time"].array,
        missing=2,
        minlen=5,
    )
    print(blinks)
