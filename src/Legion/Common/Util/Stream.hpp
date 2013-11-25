
#ifndef LEGION_COMMON_UTIL_STREAM_HPP_
#define LEGION_COMMON_UTIL_STREAM_HPP_

#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <iosfwd>

namespace legion
{
    class Color;
    class Ray;
    class Matrix;

    template<unsigned DIM, typename TYPE>
    LAPI std::ostream& operator<<( std::ostream& out,
                                   const legion::Vector<DIM, TYPE>& v ); 

    LAPI std::ostream& operator<<( std::ostream& out, const legion::Color& c ); 

    LAPI std::ostream& operator<<( std::ostream& out, const legion::Matrix& m ); 


    template<unsigned DIM, typename TYPE>
    LAPI inline std::ostream& operator<<(
        std::ostream& out,
        const legion::Vector<DIM, TYPE>& v )
    {
        out << "[";
        for( unsigned int i = 0; i < DIM-1; ++i ) out << v[i] << ", ";
        out << v[DIM-1] << "]";

        return out;
    }

}
#endif // LEGION_COMMON_UTIL_STREAM_HPP_
