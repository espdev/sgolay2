# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sgolay2


def make_surface():
    x, y = np.mgrid[-5:6:.5, -5:6:.5]
    z = y * np.sin(x) + x * np.cos(y)
    zn = z + + np.random.randn(*x.shape) * 2.
    return x, y, z, zn


def smooth_surface(z, w=9, p=3):
    sg2 = sgolay2.SGolayFilter2(window_size=w, poly_order=p)
    return sg2(z)


def visualize(x, y, z, zn, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(x, y, zn, linewidths=0.5, color='r')
    ax.scatter(x, y, zn, s=5, c='r')

    ax.plot_surface(x, y, zs, linewidth=0)
    ax.plot_surface(x, y, z, color='y', linewidth=0, alpha=0.4)

    plt.show()


def main():
    np.random.seed(12345)

    x, y, z, zn = make_surface()
    zs = smooth_surface(zn, 9, 3)

    visualize(x, y, z, zn, zs)


if __name__ == '__main__':
    main()
