"""Module for dataset creation and data processing."""
from pathlib import Path
from typing import Union

import pandas as pd
from pandas import DataFrame

from multitouch_attribution.config.paths import DATA_RAW_DIR


def read_dataset(filename: Union[Path, str]) -> DataFrame:
    """Generate DataFrames from loaded CSV data.

    Raises:
        FileNotFoundError: raised when one of the required files is missing

    Returns:
        DataFrame: returns first channels data and
                                            second customer journey data
    """
    # generate filenames and paths
    JOURNEY_DATA = DATA_RAW_DIR / filename

    try:
        journey_data = pd.read_csv(
            JOURNEY_DATA,
            parse_dates=["date_served", "date_subscribed", "date_canceled"],
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Cannot find the named files. "
            "You may need to first download the required files."
        )

    return journey_data
