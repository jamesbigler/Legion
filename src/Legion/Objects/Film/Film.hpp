
#ifndef LEGION_OBJECTS_FILM_FILM_HPP_
#define LEGION_OBJECTS_FILM_FILM_HPP_


#include <Legion/Common/Math/Vector.hpp>

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

#endif // LEGION_OBJECTS_FILM_FILM_HPP_
