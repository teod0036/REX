# note: remember to run `pip install numpy matplotlib`

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def uniform_pdf(min: float, max: float):
    assert min < max

    def f(x: NDArray):
        return np.ones_like(x) / (max - min)

    return f


def uniform_samples(min: float, max: float, k: int):
    return np.random.uniform(min, max, k)


def normal_pdf(mean: float, std: float):
    def f(x: NDArray):
        scalar = 1.0 / (std * np.sqrt(2 * np.pi))
        frac = -((x - mean) ** 2) / (2 * std**2)

        return scalar * np.exp(frac)

    return f


def normal_samples(mean: float, std: float, k: int):
    return np.random.normal(mean, std, k)


# p as given in question (the "unknown" distribution):
def p(x: NDArray):
    x1 = normal_pdf(2.0, 1.0)(x)
    x2 = normal_pdf(5.0, 2.0)(x)
    x3 = normal_pdf(9.0, 1.0)(x)

    return 0.3 * x1 + 0.4 * x2 + 0.3 * x3


def resample(x_old: NDArray, weights_old: NDArray):
    k = len(weights_old)
    x, weights = x_old[weights_old > 0], weights_old[weights_old > 0]

    # handle edge case, when all weights are 0:
    if len(weights) == 0:
        return x

    # normalize weights (so CDF works out):
    weights /= np.sum(weights)
    weights_cdf = np.cumsum(weights)

    # generate k samples of [0;1)
    rs = np.random.uniform(size=k)

    # find the indicies, such that weights_cdf[indicies] <= rs:
    indices = np.searchsorted(weights_cdf, rs, side="left")

    return x[indices]


def main():
    np.random.seed(seed=1234)

    ks = [20, 100, 1000]
    min_x, max_x = 0, 15

    _, axes = plt.subplots(2, 3, figsize=(15, 8))

    q = uniform_pdf(min_x, max_x)
    for i, k in enumerate(ks):
        # sampling step:
        x = uniform_samples(min_x, max_x, k)
        # importance computation step:
        weights = p(x) / q(x)
        # resampling step:
        weights_new = resample(x, weights)
        # plot the graphs:
        show(axes[0, i], k, min_x, max_x, "with uniform(0, 15) proposal", weights_new)

    q = normal_pdf(5, 4)
    for i, k in enumerate(ks):
        # sampling step:
        x = normal_samples(5, 4, k)
        # importance computation step:
        weights = p(x) / q(x)
        # resampling step:
        weights_new = resample(x, weights)
        # plot the graphs:
        show(axes[1, i], k, min_x, max_x, "with normal(5, 4) proposal", weights_new)

    plt.tight_layout()
    plt.show()


def show(ax, k: int, min_x: float, max_x: float, desc: str, weights: NDArray):
    x_plot = np.linspace(min_x - 1, max_x, num=k)
    ax.plot(x_plot, p(x_plot), label="Target p(x)", color="black")
    ax.hist(
        weights,
        bins=50,
        density=True,
        alpha=0.6,
        label=desc,
        color="blue",
    )
    ax.legend()
    ax.set_title(f"k = {k}")


if __name__ == "__main__":
    main()
