#ifndef __S2POINTINDEX_H_
#define __S2POINTINDEX_H_

#include <tuple>

#include "pybind11/pybind11.h"

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

    s2point_index(const xt::pytensor<double, 2> &latlon_points);
    s2point_index(const xt::pytensor<float, 2> &latlon_points);

    s2point_index(const xt::pytensor<uint64, 1> &cell_ids);

    template <class T>
    py::tuple query(const xt::pytensor<T, 2> &latlon_points);

    const xt::pytensor<uint64, 1> get_cell_ids();
    const xt::pytensor<double, 2> get_latlon_points();

private:
    index_t m_index;
    const xt::pytensor<double, 2> m_latlon_points;

    template <class T>
    void add_latlon_points(const xt::pytensor<T, 2> &latlon_points);
};


s2point_index::s2point_index(const xt::pytensor<double, 2> &latlon_points)
    : m_latlon_points(latlon_points)
{
    add_latlon_points(latlon_points);
}


s2point_index::s2point_index(const xt::pytensor<float, 2> &latlon_points)
    : m_latlon_points(latlon_points)
{
    add_latlon_points(latlon_points);
}


s2point_index::s2point_index(const xt::pytensor<uint64, 1> &cell_ids)
{
    auto num_points = cell_ids.size();

    for (std::size_t i=0; i<num_points; ++i)
    {
        S2CellId c(cell_ids(i));
        m_index.Add(c.ToPoint(), static_cast<npy_intp>(i));
    }

}

template <class T>
void s2point_index::add_latlon_points(const xt::pytensor<T, 2> &latlon_points)
{
    auto shape = latlon_points.shape();

    for (auto i=0; i<shape[0]; ++i)
    {
        S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
        m_index.Add(point, static_cast<npy_intp>(i));
    }
}


template <class T>
py::tuple s2point_index::query(const xt::pytensor<T, 2> &latlon_points)
{
    const auto shape = latlon_points.shape();
    auto results_dists = xt::pytensor<double, 1>::from_shape({shape[0]});
    auto results_idx = xt::pytensor<npy_intp, 1>::from_shape({shape[0]});

    S2ClosestPointQuery<npy_intp> query(&m_index);

    for (auto i=0; i<shape[0]; ++i)
    {
        S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
        S2ClosestPointQuery<npy_intp>::PointTarget target(point);

        auto results = query.FindClosestPoint(&target);

        results_dists(i) = results.distance().radians();
        results_idx(i) = static_cast<npy_intp>(results.data());
    }

    return py::make_tuple(results_dists, results_idx);
}


const xt::pytensor<uint64, 1> s2point_index::get_cell_ids()
{
    auto cell_ids = xt::pytensor<uint64, 1>::from_shape({m_index.num_points()});

    index_t::Iterator iter;
    iter.Init(&m_index);

    std::size_t i = 0;

    for (iter.Begin(); !iter.done(); iter.Next())
    {
        cell_ids(i) = iter.id().id();
        ++i;
    }

    return cell_ids;
}


const xt::pytensor<double, 2> s2point_index::get_latlon_points()
{
    xt::pytensor<double, 2> latlon_points(m_latlon_points);

    // index_t::Iterator iter;
    // iter.Init(&m_index);

    // std::size_t i = 0;

    // for (iter.Begin(); !iter.done(); iter.Next())
    // {
    //     S2LatLng ll(iter.point());
    //     latlon_points(i, 0) = ll.lat().degrees();
    //     latlon_points(i, 1) = ll.lng().degrees();
    //     ++i;
    // }

    return latlon_points;
}

#endif // __S2POINTINDEX_H_
