"""Notebook to generate some EDA insights"""
# %%
import matplotlib.pyplot as plt
import seaborn as sns

from multitouch_attribution.data.make_dataset import read_dataset

# %% [markdown]
# # TLDR;
# This is a basic EDA step before digging deeper into the modelling workflow. This
# dataset has a lot of potential, since the data shows for example session data as well
# as some customer-specific information like segmentation allocation. <br>
# But since our main goal will be to find an algorithmic attribution based on the given
# channel visits we reduce our analysis to channels, served dates and conversion
# informations. Any additional feature which could improve the model will be left for
# future improvement.
# %%
dataset = read_dataset(filename="customer_journey.csv")

dataset.info()

dataset.head()

dataset.describe()

# %% [markdown]

# # Basic overview
# ## First insights
# - many columns with missing values $\Rightarrow$ clean data
# - most columns are categoricals with low cardinality $\Rightarrow$ OneHotEncoding
# - few boolean columns $\Rightarrow$ cast to binaries (int)

# %% [markdown]
# # Preprocessing
# ## Restrict dataset to relevant features

# %%
dataset = dataset.filter(["user_id", "date_served", "marketing_channel", "converted"])

# %% [markdown]
# ## Handle missing values
# %%
# drop all user_ids with missing values in 'converted'
dataset = dataset.dropna(subset=["converted"])

# there is also one user_id with a NaN in 'date_served'
print(dataset.query("date_served != date_served"))
dataset = dataset.query("user_id != 'a100004504'")

# cast boolean column to int
dataset = dataset.astype({"converted": "int"})

# %% [markdown]
# ## Look at channel distributions
# %% absolute conversion impact
sns.countplot(data=dataset, x="marketing_channel", hue="converted")

# %% relative converion impact
sns.catplot(data=dataset, x="marketing_channel", y="converted", kind="bar")

# %% [markdown]

# ## Take-away: Distribution of conversion

# Looks like * House Ads* is the most used channel, but has just a few conversions
# compared to *Facebook* or *Instagram*. The channel with the most relative
# impact is email. This is a tendency we would like to see in the final attribution
# as well.

# %% [markdown]
# ## Look at channel interaction over time
# %% General channel visits
df = dataset.groupby(by=["date_served"]).agg(nb_count=("user_id", "size")).reset_index()
sns.lineplot(data=df, x="date_served", y="nb_count")
plt.xticks(rotation=45)

# %% channel visits per marketing channel
df = (
    dataset.groupby(by=["date_served", "marketing_channel"])
    .agg(nb_count=("user_id", "size"))
    .reset_index()
)

sns.lineplot(data=df, x="date_served", y="nb_count", hue="marketing_channel")
plt.xticks(rotation=45)
# %% [markdown]
# ## Surprising pattern

# It is quite surprising to this pattern for the email channel as a single peak. This
# could be problematic for a ML model if we doesn't provide the served date as a
# feature, but we keep this in mind.
