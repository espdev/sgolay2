# Two-dimensional Savitzky-Golay filter

A Savitzky–Golay filter is a digital filter that can be applied to a set of digital data points for the purpose of smoothing the data, that is, to increase the precision of the data without distorting the signal tendency. ([wikipedia](https://en.wikipedia.org/wiki/Savitzky–Golay_filter))

This code implements two-dimensional Savitzky-Golay filter that can be used for smoothing surfaces or images [1, 2].

## The example of usage

```python
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sgolay2

np.random.seed(12345)

x, y = np.mgrid[-5:6:.5, -5:6:.5]
z = y * np.sin(x) + x * np.cos(y)
zn = z + np.random.randn(*x.shape) * 2.

zs = sgolay2.SGolayFilter2(window_size=9, poly_order=3)(zn)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(x, y, zn, linewidths=0.5, color='r')
ax.scatter(x, y, zn, s=5, c='r')

ax.plot_surface(x, y, zs, linewidth=0)
ax.plot_surface(x, y, z, color='y', linewidth=0, alpha=0.4)

plt.show()
```

<img width="592" alt="sgolay2_surface" src="https://user-images.githubusercontent.com/1299189/54092147-bc511b00-4399-11e9-9be2-c44ce697161e.png">


## References

1. Ratzlaff, Kenneth L.; Johnson, Jean T. (1989). "Computation of two-dimensional polynomial least-squares convolution smoothing integers". Anal. Chem. 61 (11): 1303–5. doi:10.1021/ac00186a026.
2. Krumm, John. "Savitzky–Golay filters for 2D Images". Microsoft Research, Redmond.
