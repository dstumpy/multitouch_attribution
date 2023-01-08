"""ML approach for creating an algorithmic attribution model."""
# %%
import logging

import pandas as pd

from multitouch_attribution.data.make_dataset import read_dataset

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# %% [markdonw]

# # TLDR;
# We implement an algorithmic attribution approach using a machine learning model.
# The problem is seen as a classification problem where the model tries to predict
# whether a customer converts based on their customer journey or not. <br>
# With a trained model, we can measure the impact of each channel with respect to
# the conversion probability and exactly this impact is used to mirror the final
# attribution.

# %%


def remove_rows_with_missings(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values and cast target column to binary (int).

    Args:
        dataset (pd.DataFrame): base dataset

    Returns:
        pd.DataFrame: cleaned dataset
    """
    # drop all user_ids with missing values in 'converted'
    dataset = dataset.dropna(subset=["converted"])

    # there is also one user_id with a NaN in 'date_served'
    logging.info(dataset.query("date_served != date_served"))
    dataset = dataset.query("user_id != 'a100004504'")

    # cast boolean column to int
    dataset = dataset.astype({"converted": "int"})

    return dataset


dataset = read_dataset(filename="customer_journey.csv")
dataset = dataset.filter(["user_id", "date_served", "marketing_channel", "converted"])

dataset = remove_rows_with_missings(dataset=dataset)

# %%

# we need to find those channel visits that doesn't add value to a conversion action
# since the customer already converted -> sort also by 'converted' to distinguish
# events on the same day (e.g. with one conversion and one not)
dataset = dataset.sort_values(by=["user_id", "date_served", "converted"])

# remove all channel visits after a customer has already converted
dataset = (
    dataset.assign(cum_sum=lambda x: x.groupby("user_id").converted.cumsum())
    .query("cum_sum <= 1")
    .drop(columns="cum_sum")
)

# %%
