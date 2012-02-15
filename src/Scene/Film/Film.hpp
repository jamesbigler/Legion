
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

}


}

#endif // LEGION_FILM_FILM_HPP_
