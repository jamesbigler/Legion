
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
                                  const legion::SurfaceInfo& si)
{
    out << "pos       :" << si.position         << "\n"
        << "obj_pos   :" << si.position_object  << "\n"
        << "geo_norm  :" << si.geometric_normal << "\n"
        << "shade_norm:" << si.shading_normal   << "\n"
        << "texcoord  :" << si.texcoord         << "\n"
        << "mat_id    :" << si.material_id;
        
    return out;
}
