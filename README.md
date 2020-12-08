pys2index
=========

![test status](https://github.com/benbovy/pys2index/workflows/test/badge.svg)

Python / NumPy compatible geographical index based on
[s2geometry](https://s2geometry.io).

This project doesn't provide Python wrappers for the whole `s2geometry` library.
Instead, it aims to provide some index wrappers with an API similar to
[scipy.spatial.cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html).

Build Dependencies
------------------

- C++14 compiler
- CMake
- [s2geometry](https://github.com/google/s2geometry)
- [xtensor-python](https://github.com/xtensor-stack/xtensor-python)
- [pybind11](https://github.com/pybind/pybind11)
- Python
- NumPy

Installation (from source)
--------------------------

First, clone this repository:

``` bash
$ git clone https://github.com/benbovy/pys2index
$ cd pys2index
```

You can install all the dependencies using conda (or mamba):

``` bash
$ conda install python cxx-compiler numpy s2geometry pybind11 xtensor-python cmake -c conda-forge
```

Build and install this library

``` bash
$ python -m pip install .
```

Running the tests
-----------------

Running the tests requires `pytest` (it is also available on conda-forge).

```bash
$ pytest .
```
