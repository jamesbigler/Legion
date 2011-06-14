
#ifndef LEGION_FILM_FILM_HPP_
#define LEGION_FILM_FILM_HPP_


#include <Core/Vector.hpp>

namespace legion
{

namespace Film
{
    enum PixelFilter
    {
        PIXEL_FILTER_NONE = 0,
        PIXEL_FILTER_BOX,
        PIXEL_FILTER_TENT,
        PIXEL_FILTER_CUBIC_SPLINE
    };

    void warpSampleByTentFilter       ( const Vector2& in_sample, Vector2& filtered_sample );
    void warpSampleByBoxFilter        ( const Vector2& in_sample, Vector2& filtered_sample );
    void warpSampleByCubicSplineFilter( const Vector2& in_sample, Vector2& filtered_sample );
}


}

#endif // LEGION_FILM_FILM_HPP_
