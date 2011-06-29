
#include <Util/Stream.hpp>

using namespace legion;

std::ostream& legion::operator<<( std::ostream& out, const legion::Color& c )
{
    out << "[" << c.red() << ", " << c.green() << ", " << c.blue() << "]";
    return out;
}

