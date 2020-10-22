#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "pys2index/s2pointindex.hpp"

#include "s2/s2latlng.h"
#include "s2/s2cell_id.h"

namespace py = pybind11;


class get_cell_ids
{
public:
    get_cell_ids() = default;

    xt::pytensor<uint64, 1> operator()(xt::pytensor<double, 2> &latlon_points)
    {
        auto shape = latlon_points.shape();
        auto cell_ids = xt::pytensor<uint64, 1>::from_shape({shape[0]});

        for (auto i=0; i<shape[0]; ++i)
        {
            S2CellId cell(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
            cell_ids(i) = cell.id();
        }

        return cell_ids;
    }
};


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

    py::class_<get_cell_ids> py_get_cell_ids(m, "get_cell_ids", R"pbdoc(
        Get S2 cell ids of lat/lon coordinate points.

        Parameters
        ----------
        latlon_points : array-like of shape (n_points, 2)
            2-d array of query point coordinates (latitude, longitude) in degrees.

        Returns
        -------
        cell_ids : array-like of shape (n_points)
            1-d array of S2 cell ids of each point (dtype: uint64).

    )pbdoc");
    py_get_cell_ids.def(py::init<>());
    py_get_cell_ids.def("__call__", &get_cell_ids::operator(), py::call_guard<py::gil_scoped_release>());
    py_get_cell_ids.def("__reduce__", [py_get_cell_ids](const get_cell_ids &self) {
        return py::make_tuple(py_get_cell_ids, py::tuple());
    });

}
