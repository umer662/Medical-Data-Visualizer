import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = ((df["weight"] / (df["height"]/100)**2) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.

df["cholesterol"] = np.where(df["cholesterol"] > 1, 1, 0)
df["gluc"] = np.where(df["gluc"] > 1, 1, 0)

# Draw Categorical Plot
def draw_cat_plot():

    df_melted = pd.melt(df, id_vars=["cardio"], value_vars=["active", "alco", "cholesterol", "gluc", "overweight", "smoke"])

    fig = sns.catplot(data=df_melted, x="variable", hue='value', kind="count", col="cardio").set_ylabels("total").fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Subset Data: Remove rows where diastolic pressure is higher than systolic blood pressure
    mask_bp = df['ap_lo'] <= df['ap_hi']

    # Subset Data: Remove outliers from "height" column
    low, high = df["height"].quantile([0.025, 0.975])
    mask_height = df["height"].between(low, high, inclusive='both')
  
    # Subset Data: Remove outliers from "weight" column
    low, high = df["weight"].quantile([0.025, 0.975])
    mask_weight = df["weight"].between(low, high, inclusive='both')

    # Clean the data
    df_heat = df[mask_bp & mask_height & mask_weight]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (12,9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, vmax=0.3, square=True,
    # fmt=".1f" controls the format of the annotation
      fmt=".1f", annot=True)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig