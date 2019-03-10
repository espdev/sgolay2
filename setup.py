# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='sgolay2',
    version='0.1.0',
    py_modules=['sgolay2'],
    install_requires=[
        'numpy>=1.15.0',
        'scipy>=1.1.0',
    ],
    tests_require=['pytest'],
    url='',
    license='MIT',
    author='Eugene Prilepin',
    author_email='esp.home@gmail.com',
    description='Two-dimensional Savitzky-Golay filter'
)
