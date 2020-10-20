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
    )pbdoc";

    py::class_<s2point_index>(m, "S2PointIndex")
        .def(py::init<const xt::pytensor<double, 2>&>())
        .def(py::init<const xt::pytensor<float, 2>&>())
        .def("query", &s2point_index::query<double>, "")
        .def("query", &s2point_index::query<float>, "");

}
