import pandas as pd
import numpy as np

# load the long-form dataset
input_path = "../../data/interim/merged_formants_perceptive.csv"
df = pd.read_csv(input_path)

# columns for output
meta_cols = ["vowel_id", "word", "artist", "song", "aae_realization"]
n_timepoints = 100
f1_cols = [f"f1_{i:02d}" for i in range(n_timepoints)]
f2_cols = [f"f2_{i:02d}" for i in range(n_timepoints)]


# function to process each token group
def process_token(group):
    # sort by measurement_time
    group = group.sort_values("measurement_time")
    # drop if fewer than 3 timepoints
    if len(group) < 3:
        return None
    # drop if all f1p or all f2p are missing
    if group["f1p"].isna().all() or group["f2p"].isna().all():
        return None
    # fill individual NaNs by interpolation (linear, limit_direction="both")
    group["f1p"] = group["f1p"].interpolate(method="linear", limit_direction="both")
    group["f2p"] = group["f2p"].interpolate(method="linear", limit_direction="both")
    # after interpolation, if still any NaN, drop
    if group["f1p"].isna().any() or group["f2p"].isna().any():
        return None
    # normalize time to 0-1
    t = group["measurement_time"].values.astype(float)
    t_norm = (
        (t - t.min()) / (t.max() - t.min())
        if t.max() > t.min()
        else np.linspace(0, 1, len(t))
    )
    # interpolate to 100 points
    t_new = np.linspace(0, 1, n_timepoints)
    f1_interp = np.interp(t_new, t_norm, group["f1p"].values)
    f2_interp = np.interp(t_new, t_norm, group["f2p"].values)
    # build output row
    meta = {col: group[col].iloc[0] if col in group else None for col in meta_cols}
    row = {**meta}
    row.update({f1_cols[i]: f1_interp[i] for i in range(n_timepoints)})
    row.update({f2_cols[i]: f2_interp[i] for i in range(n_timepoints)})
    return row


# group by vowel_id and process
groups = df.groupby("vowel_id")
rows = [process_token(g) for _, g in groups]
# filter out None (dropped tokens)
rows = [r for r in rows if r is not None]

# create wide-form dataframe
df_wide = pd.DataFrame(rows)

# save to csv
df_wide.to_csv("../../data/interim/formant_wide.csv", index=False)
print(f"saved wide-form data to ../../data/interim/formant_wide.csv")
