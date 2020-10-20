#ifndef __S2POINTINDEX_H_
#define __S2POINTINDEX_H_

#include "xtensor/xtensor.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "s2/s2closest_point_query.h"
#include "s2/s2latlng.h"
#include "s2/s2point.h"
#include "s2/s2point_index.h"


class s2point_index
{
public:
    s2point_index(const xt::pytensor<double, 2> &latlon_points);
    s2point_index(const xt::pytensor<float, 2> &latlon_points);

    template <class T>
    xt::pytensor<npy_intp, 1> query(const xt::pytensor<T, 2> &latlon_points);

private:
    S2PointIndex<npy_intp> m_index;

    template <class T>
    void add_latlon_points(const xt::pytensor<T, 2> &latlon_points);
};


s2point_index::s2point_index(const xt::pytensor<double, 2> &latlon_points)
{
    add_latlon_points(latlon_points);
}


s2point_index::s2point_index(const xt::pytensor<float, 2> &latlon_points)
{
    add_latlon_points(latlon_points);
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
xt::pytensor<npy_intp, 1> s2point_index::query(const xt::pytensor<T, 2> &latlon_points)
{
    const auto shape = latlon_points.shape();
    auto results_idx = xt::pytensor<npy_intp, 1>::from_shape({shape[0]});

    S2ClosestPointQuery<npy_intp> query(&m_index);

    for (auto i=0; i<shape[0]; ++i)
    {
        S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
        S2ClosestPointQuery<npy_intp>::PointTarget target(point);

        auto results = query.FindClosestPoint(&target);

        results_idx(i) = static_cast<npy_intp>(results.data());
    }

    return results_idx;
}



#endif // __S2POINTINDEX_H_
