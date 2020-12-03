#include <memory>

#include "pybind11/pybind11.h"

#include "xtensor/xview.hpp"

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

    py::class_<s2point_index> py_s2pointindex (m, "S2PointIndex", R"pbdoc(
        S2 index for fast geographic point problems.

        Parameters
        ----------
        latlon_points : ndarray of shape (n_points, 2)
            2-d array of point coordinates (latitude, longitude) in degrees.
    )pbdoc");

    py_s2pointindex.def(py::init<const xt::pytensor<double, 2>&>(), py::call_guard<py::gil_scoped_release>());
    py_s2pointindex.def(py::init<const xt::pytensor<float, 2>&>(), py::call_guard<py::gil_scoped_release>());
    py_s2pointindex.def(py::init<const xt::pytensor<uint64, 1>&>(), py::call_guard<py::gil_scoped_release>());

    py_s2pointindex.def("query", &s2point_index::query<double>,
                        py::call_guard<py::gil_scoped_release>(),
                        R"pbdoc(
        Query the index for nearest neighbors.

        Parameters
        ----------
        latlon_points : ndarray of shape (n_points, 2), dtype=double
            2-d array of query point coordinates (latitude, longitude) in degrees.

        Returns
        -------
        distances : ndarray of shape (n_points,), dtype=double
            Distance to the nearest neighbor of the cooresponding points (in degrees).
        positions : ndarray of shape (n_points,), dtype=int
            Indices of the nearest neighbor of the corresponding points.

    )pbdoc");
    py_s2pointindex.def("query", &s2point_index::query<float>, py::call_guard<py::gil_scoped_release>(),
        "Query the index for nearest neighbors (float version).");

    py_s2pointindex.def("get_cell_ids", &s2point_index::get_cell_ids,
                        py::call_guard<py::gil_scoped_release>(),
                        py::return_value_policy::move);

    py_s2pointindex.def(py::pickle(
        [](s2point_index &idx) {
            return idx.get_cell_ids();
        },
        [](xt::pytensor<uint64, 1> &cell_ids) {
            return std::make_unique<s2point_index>(s2point_index(cell_ids));
        }
     ));

}
