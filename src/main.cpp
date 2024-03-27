//#include <cmath>
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

    py_s2pointindex.def_static("to_cell_ids",
                               &pys2::s2point_index::to_cell_ids<double>,
                               R"pbdoc(
        Convert latlon points to S2 IDs.

        Parameters
        ----------
        latlon_points : ndarray of shape (n_points, 2), dtype=double
            2-d array of point coordinates (latitude, longitude) in degrees.

        Returns
        -------
        cell_ids : ndarray of shape (n_points,), dtype=uint64
            array of cell ids.

    )pbdoc"
                              );
    py_s2pointindex.def_static("to_cell_ids",
                               &pys2::s2point_index::to_cell_ids<float>,
                               "Convert latlon points to S2 IDs (float version).");
    
    py_s2pointindex.def("query",
                        &pys2::s2point_index::query<double>,
                        R"pbdoc(
        Query the index for nearest neighbors.

        Parameters
        ----------
        latlon_points : ndarray of shape (n_points, 2), dtype=double
            2-d array of query point coordinates (latitude, longitude) in degrees.
        max_results : int
            Maximum number of nearest neighbors to return
        max_error : double
            Specifies that points up to max_error further away than the true
            closest points may be substituted in the result set (in degrees).
        max_distance : double
            Specifies that only points whose distance to the target is less than
            "max_distance" should be returned (in degrees).

        Returns
        -------
        distances : ndarray of shape (n_points,) if k=1 else (n_points, k), dtype=double
            Distance to the nearest neighbor of the cooresponding points (in degrees).
        positions : ndarray of shape (n_points,) if k=1 else (n_points, k), dtype=int
            Indices of the nearest neighbor of the corresponding points.

    )pbdoc",
                        py::arg("latlon_points"),
                        py::arg("max_results") = 1,
                        py::arg("max_error") = 0., 
                        py::arg("max_distance") = std::numeric_limits<double>::infinity());
    py_s2pointindex.def("query",
                        &pys2::s2point_index::query<float>,
                        "Query the index for nearest neighbors (float version).",
                        py::arg("latlon_points"),
                        py::arg("max_results") = 1,
                        py::arg("max_error") = 0.,
                        py::arg("max_distance") = std::numeric_limits<double>::infinity());

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
