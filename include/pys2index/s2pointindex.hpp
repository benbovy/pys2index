#ifndef __S2POINTINDEX_H_
#define __S2POINTINDEX_H_

#include <tuple>
#include <cmath>

#include "pybind11/pybind11.h"

#include "xtensor/xbuilder.hpp"
#include "xtensor/xtensor.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xview.hpp"

#if __has_include("tbb/parallel_for.h")
#define S2POINTINDEX_TBB
#include "tbb/parallel_for.h"
#endif

#include "s2/s2cell_id.h"
#include "s2/s2closest_point_query.h"
#include "s2/s2latlng.h"
#include "s2/s2point.h"
#include "s2/s2point_index.h"
#include "s2/s1chord_angle.h"

namespace py = pybind11;


namespace pys2index
{

    class s2point_index
    {
    public:
        using index_type = S2PointIndex<npy_intp>;
        using cell_ids_type = xt::pytensor<uint64, 1>;

        // the dimension is not known for query results, so use pyarray
        using positions_type = xt::pyarray<npy_intp>;

        template <class T>
        using distances_type = xt::pyarray<T>;

        template <class T>
        using query_return_type = std::tuple<distances_type<T>, positions_type>;

        template <class T>
        using points_type = xt::pytensor<T, 2>;

        s2point_index() = default;

        template <class T>
        static std::unique_ptr<s2point_index> from_points(const points_type<T>& latlon_points);

        static std::unique_ptr<s2point_index> from_cell_ids(const cell_ids_type& cell_ids);

        template <class T>
        query_return_type<T> query(const points_type<T>& latlon_points,
                                   const std::size_t max_results,
                                   const double max_error,
                                   const double max_distance);

        cell_ids_type get_cell_ids();

        template <class T>
        static cell_ids_type to_cell_ids(const points_type<T>& latlon_points);

    private:
        index_type m_index;
        cell_ids_type m_cell_ids;

        template <class T>
        void insert_latlon_points(const points_type<T>& latlon_points);

        void insert_cell_ids();
    };

    template <class T>
    std::unique_ptr<s2point_index> s2point_index::from_points(const points_type<T>& latlon_points)
    {
        auto index = std::make_unique<s2point_index>();

        index->insert_latlon_points(latlon_points);

        return index;
    }

    std::unique_ptr<s2point_index> s2point_index::from_cell_ids(const cell_ids_type& cell_ids)
    {
        auto index = std::make_unique<s2point_index>();

        index->m_cell_ids = cell_ids;
        index->insert_cell_ids();

        return index;
    }

    template <class T>
    auto s2point_index::to_cell_ids(const points_type<T>& latlon_points) -> cell_ids_type
    {
        auto n_points = latlon_points.shape()[0];
        auto cell_ids = cell_ids_type::from_shape({ n_points });

        py::gil_scoped_release release;
#ifdef S2POINTINDEX_TBB
        // indexing is very fast so we should not allow too small chunks
        std::size_t chuncksize = 1024;
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, n_points, chuncksize),
            [&](tbb::blocked_range<std::size_t> r)
            {
                for (std::size_t i = r.begin(); i < r.end(); ++i)
#else
        for (std::size_t i = 0; i < n_points; ++i)
#endif
                {
                    S2CellId c(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
                    cell_ids(i) = c.id();
                }
#ifdef S2POINTINDEX_TBB
            });
#endif
        py::gil_scoped_acquire acquire;
        return cell_ids;
    }

    template <class T>
    void s2point_index::insert_latlon_points(const points_type<T>& latlon_points)
    {
        m_cell_ids = s2point_index::to_cell_ids(latlon_points);
        this->insert_cell_ids();
    }

    void s2point_index::insert_cell_ids()
    {
        auto num_points = m_cell_ids.size();

        py::gil_scoped_release release;

        for (std::size_t i = 0; i < num_points; ++i)
        {
            S2CellId c(m_cell_ids(i));
            m_index.Add(c.ToPoint(), static_cast<npy_intp>(i));
        }

        py::gil_scoped_acquire acquire;
    }

    template <class T>
    auto s2point_index::query(const points_type<T>& latlon_points,
                              const std::size_t max_results,
                              const double max_error,
                              const double max_distance) -> query_return_type<T>
    {
        auto n_points = latlon_points.shape()[0];
        auto distances = distances_type<T>::from_shape({ n_points, max_results });
        auto positions = positions_type::from_shape({ n_points, max_results });

        py::gil_scoped_release release;
#ifndef S2POINTINDEX_TBB
        distances.fill(std::numeric_limits<T>::infinity());
        positions.fill(n_points);
#else
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n_points),
                          [&](tbb::blocked_range<std::size_t> r)
        {
            auto dview = xt::view(distances, xt::range(r.begin(), r.end()), xt::all());
            dview.fill(std::numeric_limits<T>::infinity());
            auto pview = xt::view(positions, xt::range(r.begin(), r.end()), xt::all());
            pview.fill(n_points);
#endif
        S2ClosestPointQuery<npy_intp> query(&m_index);
        query.mutable_options()->set_max_results(max_results);
        if (!std::isinf(max_distance))
        {
            query.mutable_options()->set_max_distance(S1ChordAngle::Degrees(max_distance));
        }
        if (max_error > 0.0)
        {
            query.mutable_options()->set_max_error(S1ChordAngle::Degrees(max_error));
        }
#ifdef S2POINTINDEX_TBB
        for (std::size_t i = r.begin(); i < r.end(); ++i)
#else
            for (std::size_t i = 0; i < n_points; ++i)
#endif
        {
            S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
            S2ClosestPointQuery<npy_intp>::PointTarget target(point);
            std::size_t n = 0;
            for (const auto& result : query.FindClosestPoints(&target))
            {
                distances(i, n) = static_cast<T>(result.distance().degrees());
                positions(i, n) = static_cast<npy_intp>(result.data());
                n++;
            }
        }
#ifdef S2POINTINDEX_TBB
    });
#endif
    py::gil_scoped_acquire acquire;

    return std::make_tuple(std::move(xt::squeeze(distances)), std::move(xt::squeeze(positions)));
}

auto
s2point_index::get_cell_ids() -> cell_ids_type
{
    cell_ids_type cell_ids(m_cell_ids);

    return cell_ids;
}
}


#endif  // __S2POINTINDEX_H_
