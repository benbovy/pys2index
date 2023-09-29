# pys2index

[![test status](https://github.com/benbovy/pys2index/workflows/test/badge.svg)](https://github.com/benbovy/pys2index/actions)

Python / NumPy compatible geographical index based on
[s2geometry](https://s2geometry.io).

This project doesn't provide Python wrappers for the whole `s2geometry` library.
Instead, it aims to provide some index wrappers with an API similar to
[scipy.spatial.cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html).

## Build Dependencies

- C++17 compiler
- CMake
- [s2geometry](https://github.com/google/s2geometry)
- [scikit-build-core](https://github.com/scikit-build/scikit-build-core)
- [xtensor-python](https://github.com/xtensor-stack/xtensor-python)
- [pybind11](https://github.com/pybind/pybind11)
- Python
- NumPy

## Installation

### Using conda

pys2index can be installed using [conda](https://docs.conda.io) (or
[mamba](https://github.com/mamba-org/mamba)):

``` bash
$ conda install pys2index -c conda-forge
```

### From source

First, clone this repository:

``` bash
$ git clone https://github.com/benbovy/pys2index
$ cd pys2index
```

You can install all the dependencies using conda (or mamba):

``` bash
$ conda install python cxx-compiler numpy s2geometry pybind11 xtensor-python cmake scikit-build-core -c conda-forge
```

Build and install this library

``` bash
$ python -m pip install . --no-build-isolation
```

## Usage

```python
In [1]: import numpy as np

In [2]: from pys2index import S2PointIndex

In [3]: latlon_points = np.array([[40.0, 15.0],
   ...:                           [-20.0, 53.0],
   ...:                           [81.0, 153.0]])
   ...:

In [4]: index = S2PointIndex(latlon_points)

In [5]: query_points = np.array([[-10.0, 60.0],
   ...:                          [45.0, -20.0],
   ...:                          [75.0, 122.0]])
   ...:

In [6]: distances, positions = index.query(query_points)

In [7]: distances
Out[7]: array([12.06534671, 26.07312392,  8.60671311])

In [8]: positions
Out[8]: array([1, 0, 2])

In [9]: index.get_cell_ids()
Out[9]: array([1386017682036854979, 2415595305706115691, 6525033740530229539],
              dtype=uint64)

```

## Running the tests

Running the tests requires `pytest` (it is also available on conda-forge).

```bash
$ pytest .
```
