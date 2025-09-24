# note: remember to run `pip install numpy matplotlib`

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

values = [
    1307.59,
    1297.24,
    1282.76,
    1260.00,
    1266.21,
    1251.72,
    1237.24,
    1241.38,
    1231.03,
    1191.72,
]

plt.hist(
    values,
    bins=5,
    density=True,
    alpha=0.6,
    label="focal length (pixels)",
    color="blue",
)

mu, std = norm.fit(values)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)


plt.xlabel("Z (mm)")
plt.ylabel("focal length (pixels)")
plt.show()
