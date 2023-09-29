#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "pys2index/s2pointindex.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace pys2 = pys2index;


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
           S2PointIndex.get_cell_ids
    )pbdoc";

    py::class_<pys2::s2point_index> py_s2pointindex(m, "S2PointIndex", R"pbdoc(
        S2 index for fast geographic point problems.

        Parameters
        ----------
        latlon_points : ndarray of shape (n_points, 2)
            2-d array of point coordinates (latitude, longitude) in degrees.
    )pbdoc");

    py_s2pointindex.def(py::init(&pys2::s2point_index::from_points<double>));
    py_s2pointindex.def(py::init(&pys2::s2point_index::from_points<float>));
    py_s2pointindex.def(py::init(&pys2::s2point_index::from_cell_ids));

    py_s2pointindex.def("query",
                        &pys2::s2point_index::query<double>,
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
    py_s2pointindex.def("query",
                        &pys2::s2point_index::query<float>,
                        "Query the index for nearest neighbors (float version).");

    py_s2pointindex.def(
        "get_cell_ids", &pys2::s2point_index::get_cell_ids, py::return_value_policy::move);

    py_s2pointindex.def(py::pickle([](pys2::s2point_index& idx) { return idx.get_cell_ids(); },
                                   [](pys2::s2point_index::cell_ids_type& cell_ids)
                                   { return pys2::s2point_index::from_cell_ids(cell_ids); }));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
