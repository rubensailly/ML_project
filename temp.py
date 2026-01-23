import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 0) Fix notebook syntax issues first (remove stray 'd' cell content and bad indentation)

# 1) Clean target BEFORE anything else (y must be numeric)
df_train = PreCleaner.cleanTarget(df_train, "Time_taken(min)")

# 2) Build a cleaned feature table for EDA (do NOT include target in transformer)
X_clean = cleaner.transform(df_train.drop(columns=["Time_taken(min)"]))
y = df_train["Time_taken(min)"].astype(float)

df_eda = X_clean.copy()
df_eda["Time_taken(min)"] = y

# -------------------------
# A) Basic EDA: missingness, dtypes, descriptive stats
# -------------------------
missing = (df_eda.isna().mean().sort_values(ascending=False) * 100).to_frame("missing_%")
display(missing.head(20))

num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df_eda.columns if c not in num_cols]

display(df_eda[num_cols].describe().T)

# -------------------------
# B) Target correlation (bivariate) for numeric variables
# -------------------------
corr = df_eda[num_cols].corr(numeric_only=True)["Time_taken(min)"].sort_values(key=lambda s: s.abs(), ascending=False)
display(corr.head(20))

plt.figure(figsize=(8, 6))
sns.barplot(x=corr.drop("Time_taken(min)").head(15).values, y=corr.drop("Time_taken(min)").head(15).index, orient="h")
plt.title("Top numeric correlations with target (abs-sorted)")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Scatter for the strongest few
top_feats = corr.drop("Time_taken(min)").head(4).index
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()
for ax, col in zip(axes, top_feats):
    sns.scatterplot(data=df_eda, x=col, y="Time_taken(min)", alpha=0.2, ax=ax)
    ax.set_title(f"{col} vs target")
plt.tight_layout()
plt.show()

# -------------------------
# C) Categorical distributions + relationship with target
# -------------------------
for col in cat_cols:
    vc = df_eda[col].value_counts(dropna=False).head(15)
    display(vc.to_frame("count").T)

# For low-cardinality categoricals: boxplot vs target
low_card_cols = [c for c in cat_cols if df_eda[c].nunique(dropna=False) <= 15]
for col in low_card_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df_eda, x=col, y="Time_taken(min)")
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Target distribution by {col}")
    plt.tight_layout()
    plt.show()

# -------------------------
# D) Outlier detection: prefer quantile capping (winsorization) on numeric features
# -------------------------
def clip_by_quantiles(df, cols, low=0.01, high=0.99):
    df2 = df.copy()
    bounds = {}
    for c in cols:
        if c == "Time_taken(min)":
            continue
        lo = df2[c].quantile(low)
        hi = df2[c].quantile(high)
        bounds[c] = (lo, hi)
        df2[c] = df2[c].clip(lo, hi)
    return df2, bounds

# Optional: visualize boxplots before/after for a few key numeric columns
key_num = [c for c in ["Computed_distance", "Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"] if c in df_eda.columns]
plt.figure(figsize=(10, 4))
sns.boxplot(data=df_eda[key_num])
plt.title("Before capping: numeric boxplots")
plt.tight_layout()
plt.show()

df_capped, cap_bounds = clip_by_quantiles(df_eda, num_cols, low=0.01, high=0.99)

plt.figure(figsize=(10, 4))
sns.boxplot(data=df_capped[key_num])
plt.title("After capping (1%/99%): numeric boxplots")
plt.tight_layout()
plt.show()

# Notes:
# - Do not cap the target unless you have a clear reason; if you do, evaluate impact with CV.
# - Instead of dropping rows, capping keeps dataset size and stability.

# -------------------------
# E) Feature engineering ideas (beyond distance)
# -------------------------
# If you want more from time features (already created in PreCleaner):
# - delivery prep time in minutes
if "Order_Picked_Time_Min" in df_eda.columns and "Order_Time_Min" in df_eda.columns:
    df_eda["Prep_time_min"] = (df_eda["Order_Picked_Time_Min"] - df_eda["Order_Time_Min"]).clip(lower=0)

# - cyclical encoding for order time (captures day cycle)
if "Order_Time_Min" in df_eda.columns:
    minutes_in_day = 24 * 60
    df_eda["Order_Time_sin"] = np.sin(2 * np.pi * df_eda["Order_Time_Min"] / minutes_in_day)
    df_eda["Order_Time_cos"] = np.cos(2 * np.pi * df_eda["Order_Time_Min"] / minutes_in_day)

display(df_eda[["Time_taken(min)"] + [c for c in ["Computed_distance", "Prep_time_min", "Order_Time_sin", "Order_Time_cos"] if c in df_eda.columns]].head())