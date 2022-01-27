#ifndef BOX_H_
#define BOX_H_

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "s2/s2latlng.h"
#include "s2/s2latlng_rect.h"


namespace py = pybind11;


namespace pys2index
{

    class box
    {
    public:
        box() = default;
        box(double lon_min, double lat_min, double lon_max, double lat_max)
        {
            auto lo = S2LatLng::FromDegrees(lat_min, lon_min);
            auto hi = S2LatLng::FromDegrees(lat_max, lon_max);
            m_s2obj = S2LatLngRect(lo, hi);
        }

        bool contains(const box &other) const
        {
            return m_s2obj.Contains(other.m_s2obj);
        }

    private:
        S2LatLngRect m_s2obj;
    };


    bool contains(const box &a, const box &b)
    {
        return a.contains(b);
    }

    bool dummy(const double a)
    {
        xt::pytensor<box, 1> test;
        return a > 1.0;
    }

    xt::pytensor<box, 1> create_box_array()
    {
        xt::pytensor<box, 1> arr({2});

        arr(0) = box(0., 45., 5., 48.);
        arr(1) = box(0., 45., 5., 48.);

        return arr;
    }

    bool contains_alt(double a_lon_min, double a_lat_min, double a_lon_max, double a_lat_max,
                      double b_lon_min, double b_lat_min, double b_lon_max, double b_lat_max)
    {
        auto a_box = box(a_lon_min, a_lat_min, a_lon_max, a_lat_max);
        auto b_box = box(b_lon_min, b_lat_min, b_lon_max, b_lat_max);
        return a_box.contains(b_box);
    }

    xt::pytensor<bool, 1> contains_arr1d(const xt::pytensor<box, 1> &a, const xt::pytensor<box, 1> &b)
    {
        xt::pytensor<bool, 1> res(a.shape());

        for (std::size_t i=0; i<a.size(); ++i)
        {
            res(i) = a(i).contains(b(i));
        }

        return res;

    }

}


// Try registering box as a valid numpy object dtype recognized
// by pybind11
// See: https://github.com/pybind/pybind11/issues/1776#issuecomment-492230679
namespace pybind11 { namespace detail {

// template <typename T>
// struct npy_scalar_caster {
//   PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
//   using Array = array_t<T>;

//   bool load(handle src, bool convert) {
//     // Taken from Eigen casters. Permits either scalar dtype or scalar array.
//     handle type = dtype::of<T>().attr("type");  // Could make more efficient.
//     if (!convert && !isinstance<Array>(src) && !isinstance(src, type))
//       return false;
//     Array tmp = Array::ensure(src);
//     if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
//       this->value = *tmp.data();
//       return true;
//     }
//     return false;
//   }

//   static handle cast(T src, return_value_policy, handle) {
//     Array tmp({1});
//     tmp.mutable_at(0) = src;
//     tmp.resize({});
//     // You could also just return the array if you want a scalar array.
//     object scalar = tmp[tuple()];
//     return scalar.release();
//   }
// };

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.object).num)'
//constexpr int NPY_OBJECT = 17;

// Kinda following: https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <>
struct npy_format_descriptor<pys2index::box> {
    static constexpr auto name = _("object");
    enum { value = npy_api::NPY_OBJECT_ };
    static pybind11::dtype dtype()
    {
        if (auto ptr = npy_api::get().PyArray_DescrFromType_(value))
        {
            return reinterpret_borrow<pybind11::dtype>(ptr);
        }
        pybind11_fail("Unsupported buffer format!");
    }
};

// template <>
// struct type_caster<pys2index::box> : npy_scalar_caster<pys2index::box> {
//   static constexpr auto name = _("object");
// };

}}  // namespace pybind11::detail



#endif // BOX_H_
