from mssdppg.wind.weibull import bin_probabilities


def test_weibull_bins_sum_to_one():
    speeds = list(range(0, 26))
    bins = bin_probabilities(speeds, k=2.0, c=7.5)
    total = sum(prob for _, prob in bins)
    assert abs(total - 1.0) < 1e-6
