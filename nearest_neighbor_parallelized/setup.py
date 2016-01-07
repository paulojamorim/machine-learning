from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


ext_modules = [Extension("nearest_neighbor_parallelized", ["nearest_neighbor_parallelized.pyx"],
                         language='c++',
                         extra_compile_args=['-std=c++11','-fopenmp'],
                         extra_link_args=['-std=c++11','-fopenmp'],), ]



setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)

"""

ext_modules = [Extension("nearest_neighbor_parallelized", ["nearest_neighbor_parallelized.pyx"],
                         language='c++',
                         extra_compile_args=['-std=c++11','-fopenmp'],
                         extra_link_args=['-std=c++11','-fopenmp'],), ]

"""


"""

ext_modules = [Extension("nearest_neighbor_parallelized", ["nearest_neighbor_parallelized.pyx"],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'],), ]
"""
