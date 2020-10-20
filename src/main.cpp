#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "pys2index/s2pointindex.hpp"

namespace py = pybind11;


PYBIND11_MODULE(pys2index, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        Python/NumPy compatible geographical index based on s2geometry

        .. currentmodule:: pys2index

        .. autosummary::
           :toctree: _generate

           S2PointIndex
           S2PointIndex.query
    )pbdoc";

    py::class_<s2point_index>(m, "S2PointIndex", R"pbdoc(
            S2 index for fast geographic point problems.

            Parameters
            ----------
            latlon_points : array-like of shape (n_points, 2)
                2-d array of point coordinates (latitude, longitude) in degrees.
        )pbdoc")
        .def(py::init<const xt::pytensor<double, 2>&>())
        .def(py::init<const xt::pytensor<float, 2>&>())
        .def("query", &s2point_index::query<double>, R"pbdoc(
            Query the index for nearest neighbors.

            Parameters
            ----------
            latlon_points : array-like of shape (n_points, 2)
                2-d array of query point coordinates (latitude, longitude) in degrees.
        )pbdoc")
        .def("query", &s2point_index::query<float>,
             "Query the index for nearest neighbors (float version).");

}
