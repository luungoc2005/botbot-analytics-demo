import numpy as np

def cumulative_sum_threshold(values, percentile):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def normalize_scale(attr, scale_factor):
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, skipping normalization."
            "This likely means that attribution values are all close to 0."
        )
        return np.clip(attr, -1, 1)
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def normalize_attr(attr, outlier_perc=2):
    attr_combined = np.sum(attr, axis=2)

    threshold = cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)

    return normalize_scale(attr_combined, threshold)

def get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)