#ifndef __S2POINTINDEX_H_
#define __S2POINTINDEX_H_

#include <tuple>

#include "pybind11/pybind11.h"

#include "xtensor/xbuilder.hpp"
#include "xtensor/xtensor.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "s2/s2cell_id.h"
#include "s2/s2closest_point_query.h"
#include "s2/s2latlng.h"
#include "s2/s2point.h"
#include "s2/s2point_index.h"

namespace py = pybind11;


class s2point_index
{
public:
    using index_t = S2PointIndex<npy_intp>;

    s2point_index(const s2point_index &idx);

    s2point_index(const xt::pytensor<double, 2> &latlon_points);
    s2point_index(const xt::pytensor<float, 2> &latlon_points);

    s2point_index(const xt::pytensor<uint64, 1> &cell_ids);

    template <class T>
    py::tuple query(const xt::pytensor<T, 2> &latlon_points);

    const xt::pytensor<uint64, 1> get_cell_ids();
    const xt::pytensor<double, 2> get_latlon_points();

private:
    index_t m_index;
    xt::xtensor<uint64, 1> m_cell_ids;

    template <class T>
    void insert_latlon_points(const xt::pytensor<T, 2> &latlon_points);

    void insert_cell_ids();
};


s2point_index::s2point_index(const s2point_index &idx)
    : m_cell_ids(idx.m_cell_ids)
{
    insert_cell_ids();
}


s2point_index::s2point_index(const xt::pytensor<double, 2> &latlon_points)
{
    insert_latlon_points(latlon_points);
}


s2point_index::s2point_index(const xt::pytensor<float, 2> &latlon_points)
{
    insert_latlon_points(latlon_points);
}


s2point_index::s2point_index(const xt::pytensor<uint64, 1> &cell_ids)
    : m_cell_ids(cell_ids)
{
    insert_cell_ids();
}


void s2point_index::insert_cell_ids()
{
    auto num_points = m_cell_ids.size();

    for (std::size_t i=0; i<num_points; ++i)
    {
        S2CellId c(m_cell_ids(i));
        m_index.Add(c.ToPoint(), static_cast<npy_intp>(i));
    }

}


template <class T>
void s2point_index::insert_latlon_points(const xt::pytensor<T, 2> &latlon_points)
{
    auto n_points = latlon_points.shape()[0];
    m_cell_ids.resize({static_cast<std::size_t>(n_points)});

    for (auto i=0; i<n_points; ++i)
    {
        S2CellId c(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));

        m_cell_ids(i) = c.id();
        m_index.Add(c.ToPoint(), static_cast<npy_intp>(i));
    }

}


template <class T>
py::tuple s2point_index::query(const xt::pytensor<T, 2> &latlon_points)
{
    auto n_points = latlon_points.shape()[0];
    auto distances = xt::pytensor<double, 1>::from_shape({n_points});
    auto positions = xt::pytensor<npy_intp, 1>::from_shape({n_points});

    S2ClosestPointQuery<npy_intp> query(&m_index);

    for (auto i=0; i<n_points; ++i)
    {
        S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
        S2ClosestPointQuery<npy_intp>::PointTarget target(point);

        auto results = query.FindClosestPoint(&target);

        distances(i) = results.distance().degrees();
        positions(i) = static_cast<npy_intp>(results.data());
    }

    return py::make_tuple(std::move(distances), std::move(positions));
}


const xt::pytensor<uint64, 1> s2point_index::get_cell_ids()
{
    xt::pytensor<uint64, 1> cell_ids(m_cell_ids);

    return cell_ids;
}


#endif // __S2POINTINDEX_H_
