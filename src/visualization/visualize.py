import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# import df
df = pd.read_csv("../../data/processed/formant_features.csv")

# get features
features = df.columns.tolist()

# drop first two columns
features = features[2:]
# select f1 features
f1_features = [f for f in features if f.startswith("f1")]

# select f2 features
f2_features = [f for f in features if f.startswith("f2")]

################
# UMAP projection #
###############

# use umap to reduce dimensionality of formant_wide data and plot
wide_form = pd.read_csv("../../data/interim/formant_wide.csv")

# select only f1/f2 columns for UMAP
f1_cols = [col for col in wide_form.columns if col.startswith("f1_")]
f2_cols = [col for col in wide_form.columns if col.startswith("f2_")]
formant_cols = f1_cols + f2_cols

X_wide = wide_form[formant_cols].values

# fit UMAP
umap_model = umap.UMAP(random_state=42)
X_umap_wide = umap_model.fit_transform(X_wide)

# add to dataframe for plotting
wide_form["UMAP1"] = X_umap_wide[:, 0]
wide_form["UMAP2"] = X_umap_wide[:, 1]

# create custom color mapping for 0/1
label_map = {1: "monophthong", 0: "diphthong"}
colors = ["#3373A1", "#E1812C"]

# plot
plt.figure(figsize=(8, 6))
for i, (label, name) in enumerate(label_map.items()):
    idx = wide_form["aae_realization"] == label
    plt.scatter(
        wide_form.loc[idx, "UMAP1"],
        wide_form.loc[idx, "UMAP2"],
        c=colors[i],
        label=name,
        alpha=0.7,
        edgecolor="none",
    )

plt.title("UMAP projection")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(title="", loc="best")
plt.tight_layout()
plt.savefig("../../figures/umap_projection.png", dpi=600)

#################
# Euclidean distance #
#################

plt.figure(figsize=(8, 6))
sns.violinplot(
    hue="perceptive_label",
    y="euclid_dist",
    data=df,
    split=True,
    inner="box",
    linewidth=2,
    legend=1,
)
# add horizontal line at y=0
plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
plt.title("Euclidean distance")
plt.ylabel("")
plt.tight_layout()
plt.legend(loc="upper left")
plt.savefig("../../figures/euclidean_distance.png", dpi=600)


###################
# F1 features violin plot #
###################

# calculate grid dimensions
n_features = len(f1_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

# create subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

# plot each feature
for i, feat in enumerate(f1_features):
    # only show legend for first plot
    show_legend = i == 0
    sns.violinplot(
        hue="perceptive_label",
        y=feat,
        data=df,
        split=True,
        inner="box",
        linewidth=2,
        ax=axes[i],
        legend=show_legend,
    )
    axes[i].set_title(f"{feat}".replace("_", " ").capitalize())
    axes[i].set_ylabel("")

    # add horizontal line at y=0
    axes[i].axhline(y=0, color="black", linestyle="--", alpha=0.5)

    # set custom ylim for first plot, default for others
    if i == 0:
        # custom ylim for first plot
        axes[i].set_ylim(-4, 4)
    else:
        # set ylim to 5 times iqr
        q1 = df[feat].quantile(0.25)
        q3 = df[feat].quantile(0.75)
        iqr = q3 - q1
        y_min = q1 - 2.5 * iqr
        y_max = q3 + 2.5 * iqr
        axes[i].set_ylim(y_min, y_max)

    # move legend to top left for first plot
    if i == 0:
        axes[i].legend(loc="upper left")

# hide empty subplots
for i in range(n_features, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig("../../figures/f1_features.png", dpi=600)


###################
# F2 features violin plot #
###################

# calculate grid dimensions
n_features = len(f2_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

# create subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

# plot each feature
for i, feat in enumerate(f2_features):
    # only show legend for first plot
    show_legend = i == 0
    sns.violinplot(
        hue="perceptive_label",
        y=feat,
        data=df,
        split=True,
        inner="box",
        linewidth=2,
        ax=axes[i],
        legend=show_legend,
    )
    axes[i].set_title(f"{feat}".replace("_", " ").capitalize())
    axes[i].set_ylabel("")

    # add horizontal line at y=0
    axes[i].axhline(y=0, color="black", linestyle="--", alpha=0.5)

    if i == 0:
        # custom ylim for first plot
        axes[i].set_ylim(-4, 4)
    else:
        # set ylim to 5 times iqr
        q1 = df[feat].quantile(0.25)
        q3 = df[feat].quantile(0.75)
        iqr = q3 - q1
        y_min = q1 - 2.5 * iqr
        y_max = q3 + 2.5 * iqr
        axes[i].set_ylim(y_min, y_max)

    # move legend to top left for first plot
    if i == 0:
        axes[i].legend(loc="upper left")

# hide empty subplots
for i in range(n_features, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig("../../figures/f2_features.png", dpi=600)
