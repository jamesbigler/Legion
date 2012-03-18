
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <optixu/optixu_math_stream.h>

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


std::ostream& legion::operator<<( std::ostream& out, const legion::Ray& ray )
{
    out << "( " << ray.origin() << " " << ray.direction() 
        << " " << ray.tMax() << ", " << ray.time() << " )";
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
