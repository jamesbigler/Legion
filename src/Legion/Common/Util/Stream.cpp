
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <optixu/optixu_math_stream.h>

#include <iostream>

using namespace legion;


std::ostream& legion::operator<<( std::ostream& out, const legion::Color& c )
{
    out << "[" << c.red() << ", " << c.green() << ", " << c.blue() << "]";
    return out;
}


std::ostream& legion::operator<<( std::ostream& out, const legion::Ray& ray )
{
    out << "( " << ray.getOrigin() << " " << ray.getDirection() 
        << " " << ray.tInterval() << " )";
    return out;
}
    

std::ostream& legion::operator<<( std::ostream& out,
                                  const legion::LocalGeometry& lgeom)
{
    out << "pos       :" << lgeom.position         << "\n"
        << "obj_pos   :" << lgeom.position_object  << "\n"
        << "geo_norm  :" << lgeom.geometric_normal << "\n"
        << "shade_norm:" << lgeom.shading_normal   << "\n"
        << "texcoord  :" << lgeom.texcoord         << "\n"
        << "mat_id    :" << lgeom.material_id;
        
    return out;
}
