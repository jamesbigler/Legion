
#include <Util/Stream.hpp>
#include <Core/Ray.hpp>
#include <Core/Color.hpp>


using namespace legion;

std::ostream& legion::operator<<( std::ostream& out, const legion::Color& c )
{
    out << "[" << c.red() << ", " << c.green() << ", " << c.blue() << "]";
    return out;
}


std::ostream& legion::operator<<( std::ostream& out, const legion::Ray& ray )
{
    out << "( " << ray.getOrigin() << " " << ray.getDirection() << " " << ray.tInterval() << " )";
    return out;
}
