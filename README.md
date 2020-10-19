pys2index
=========

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

Installation
------------

**On Unix (Linux, MacOS)**

 - clone this repository
 - `pip install ./pys2index`

Building the documentation
--------------------------

Documentation for the example project is generated using Sphinx. Sphinx has the
ability to automatically inspect the signatures and documentation strings in
the extension module to generate beautiful documentation in a variety formats.
The following command generates HTML-based reference documentation; for other
formats please refer to the Sphinx manual:

 - `pys2index/docs`
 - `make html`


Running the tests
-----------------

Running the tests requires `pytest`.

```bash
py.test .
```
