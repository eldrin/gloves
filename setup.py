##################################################################
#
# this setup script largely refers lots of its part to `implicit`
# (https://github.com/benfred/implicit)
#
##################################################################

import os
from os.path import join
import sys
import glob
import logging
import platform
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext


NAME = 'gloves'
VERSION = '0.0.1'

use_openmp = True

def define_extensions():
    if sys.platform.startswith('win'):
        # compile args from
        # https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        compile_args = ['/O2', '/openmp']
        link_args = []
    else:
        gcc = extract_gcc_binaries()
        if gcc is not None:
            rpath = "/usr/local/opt/gcc/lib/gcc/" + gcc[-1] + "/"
            link_args = ["-W1,-rpath," + rpath]
        else:
            link_args = []

    compile_args = ["-Wno-unused-function", "-Wno-maybe-uninitialized", "-O3", "-ffast-math"]
    if use_openmp:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

    compile_args.append("-std=c++11")
    link_args.append("-std=c++11")

    # src_ext = ".pyx"
    modules = [
        Extension(
            f"gloves.solvers.{cython_module}",
            [join("gloves", "solvers", f"{cython_module}.pyx")],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )
        for cython_module in ['_als', '_sgd']
    ]
    modules.extend([
        Extension(
            "gloves.corpus._corpus",
            [join("gloves", "corpus", "_corpus.pyx")],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args
        )
    ])

    return cythonize(modules)


# set_gcc copied from glove-python project
# https://github.com/maciejkula/glove-python

def extract_gcc_binaries():
    """Try to find GCC on OSX for OpenMP support."""
    patterns = [
        "/opt/local/bin/g++-mp-[0-9]*.[0-9]*",
        "/opt/local/bin/g++-mp-[0-9]*",
        "/usr/local/bin/g++-[0-9]*.[0-9]*",
        "/usr/local/bin/g++-[0-9]*",
    ]
    if platform.system() == "Darwin":
        gcc_binaries = []
        for pattern in patterns:
            gcc_binaries += glob.glob(pattern)
        gcc_binaries.sort()
        if gcc_binaries:
            _, gcc = os.path.split(gcc_binaries[-1])
            return gcc
        else:
            return None
    else:
        return None


def set_gcc():
    """Try to use GCC on OSX for OpenMP support."""
    # For macports and homebrew
    if platform.system() == "Darwin":
        gcc = extract_gcc_binaries()

        if gcc is not None:
            os.environ["CC"] = gcc
            os.environ["CXX"] = gcc

        else:
            global use_openmp
            use_openmp = False
            logging.warning(
                "No GCC available. Install gcc from Homebrew " "using brew install gcc."
            )


set_gcc()


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f]


setup(name=NAME,
      version=VERSION,
      description='Implementation of GloVe with different solvers',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      keywords=['GloVe', 'ALS', 'Adagrad'],
      url='http://github.com/eldrin/gloves',
      author='Jaehun Kim',
      author_email='j.h.kim@tudelft.nl',
      license='MIT',
      packages=find_packages(),
      package_data={'gloves': ['data/*.json']},
      install_requires=requirements(),
      setup_requires=["setuptools>=18.0", "Cython>=0.24"],
      extras_require={
          'dev': [],
          'opthyper': ['scikit-optimize>=0.8.1']
      },
      ext_modules=define_extensions(),
      cmdclass={"build_ext": build_ext},
      entry_points = {
          'console_scripts': [
              'cooccur=gloves.cli.cooccur:main',
              'gloves=gloves.cli.main:main'
          ],
      },
      test_suite='tests',
      zip_safe=False)
