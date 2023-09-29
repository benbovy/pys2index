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


namespace pys2index
{

    class s2point_index
    {
    public:
        using index_type = S2PointIndex<npy_intp>;
        using cell_ids_type = xt::pytensor<uint64, 1>;
        using positions_type = xt::pytensor<npy_intp, 1>;

        template <class T>
        using distances_type = xt::pytensor<T, 1>;

        template <class T>
        using query_return_type = std::tuple<distances_type<T>, positions_type>;

        template <class T>
        using points_type = xt::pytensor<T, 2>;

        s2point_index() = default;

        template <class T>
        static std::unique_ptr<s2point_index> from_points(const points_type<T>& latlon_points);

        static std::unique_ptr<s2point_index> from_cell_ids(const cell_ids_type& cell_ids);

        template <class T>
        query_return_type<T> query(const points_type<T>& latlon_points);

        cell_ids_type get_cell_ids();

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
    void s2point_index::insert_latlon_points(const points_type<T>& latlon_points)
    {
        auto n_points = latlon_points.shape()[0];
        m_cell_ids.resize({ n_points });

        py::gil_scoped_release release;

        for (auto i = 0; i < n_points; ++i)
        {
            S2CellId c(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));

            m_cell_ids(i) = c.id();
            m_index.Add(c.ToPoint(), static_cast<npy_intp>(i));
        }

        py::gil_scoped_acquire acquire;
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
    auto s2point_index::query(const points_type<T>& latlon_points) -> query_return_type<T>
    {
        auto n_points = latlon_points.shape()[0];
        auto distances = distances_type<T>::from_shape({ n_points });
        auto positions = positions_type::from_shape({ n_points });

        py::gil_scoped_release release;

        S2ClosestPointQuery<npy_intp> query(&m_index);

        for (auto i = 0; i < n_points; ++i)
        {
            S2Point point(S2LatLng::FromDegrees(latlon_points(i, 0), latlon_points(i, 1)));
            S2ClosestPointQuery<npy_intp>::PointTarget target(point);

            auto results = query.FindClosestPoint(&target);

            distances(i) = static_cast<T>(results.distance().degrees());
            positions(i) = static_cast<npy_intp>(results.data());
        }

        py::gil_scoped_acquire acquire;

        return std::make_tuple(std::move(distances), std::move(positions));
    }

    auto s2point_index::get_cell_ids() -> cell_ids_type
    {
        cell_ids_type cell_ids(m_cell_ids);

        return cell_ids;
    }

}


#endif  // __S2POINTINDEX_H_
