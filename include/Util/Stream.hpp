
#ifndef LEGION_UTIL_STREAM_HPP_
#define LEGION_UTIL_STREAM_HPP_

#include <ostream>
#include <Core/Vector.hpp>


namespace legion
{
    class Color;
    class Ray;

    template<unsigned DIM, typename TYPE>
    std::ostream& operator<<( std::ostream& out, const legion::Vector<DIM, TYPE>& v ); 

    std::ostream& operator<<( std::ostream& out, const legion::Color& c ); 

    std::ostream& operator<<( std::ostream& out, const legion::Ray& ray ); 





    template<unsigned DIM, typename TYPE>
    inline std::ostream& operator<<( std::ostream& out, const legion::Vector<DIM, TYPE>& v )
    {
        out << "[";
        for( unsigned int i = 0; i < DIM-1; ++i ) out << v[i] << ", ";
        out << v[DIM-1] << "]";

        return out;
    }

}
#endif // LEGION_UTIL_STREAM_HPP_
