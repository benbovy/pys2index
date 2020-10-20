#ifndef __S2POINTINDEX_H_
#define __S2POINTINDEX_H_

#include "xtensor/xtensor.hpp"
#include "xtensor/xbuilder.hpp"

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
    xt::pytensor<int, 1> query(const xt::pytensor<T, 2> &latlon_points);

private:
    S2PointIndex<int> m_index;

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

    for (int i=0; i<shape[0]; ++i)
    {
        S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
        m_index.Add(point, i);
    }
}


template <class T>
xt::pytensor<int, 1> s2point_index::query(const xt::pytensor<T, 2> &latlon_points)
{
    auto shape = latlon_points.shape();

    S2ClosestPointQuery<int> query(&m_index);

    xt::pytensor<int, 1> results_idx = xt::empty<int>({shape[0]});

    for (int i=0; i<shape[0]; ++i)
    {
        S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
        S2ClosestPointQuery<int>::PointTarget target(point);

        auto results = query.FindClosestPoint(&target);

        results_idx(i) = results.data();
    }

    return results_idx;
}



#endif // __S2POINTINDEX_H_
