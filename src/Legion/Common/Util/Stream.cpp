
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Color.hpp>

#include <iostream>
#include <iomanip>

using namespace legion;


std::ostream& legion::operator<<( std::ostream& out, const legion::Color& c )
{
    out << "[" 
        << c.red()   << ", " 
        << c.green() << ", " 
        << c.blue()
        << "]";
    return out;
}


std::ostream& legion::operator<<( std::ostream& out, const legion::Matrix& m )
{
    out << std::fixed 
        << "[" << m[ 0] << ", " << m[ 1] << ", " << m[ 2] << ", " << m[ 3] 
        << " " << m[ 4] << ", " << m[ 5] << ", " << m[ 6] << ", " << m[ 7] 
        << " " << m[ 8] << ", " << m[ 9] << ", " << m[10] << ", " << m[11] 
        << " " << m[12] << ", " << m[13] << ", " << m[14] << ", " << m[15]
        << "]";
    return out;
}
