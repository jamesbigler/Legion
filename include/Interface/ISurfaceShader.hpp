
#ifndef LEGION_INTERFACE_ISURFACESHADER_H_
#define LEGION_INTERFACE_ISURFACESHADER_H_


#include <Private/APIBase.hpp>
#include <Core/Color.hpp>
#include <Core/Vector.hpp>


// TODO: break this into BSDF and MaterialShader????

namespace legion
{

namespace Shader 
{
    class Geometry
    {
    };
}

class ISurfaceShader : public APIBase
{
public:
    ISurfaceShader( const std::string& name );
    virtual ~ISurfaceShader();

    virtual void  sample  ( const Vector2& seed, const Vector3& w_out, const Shader::Geometry& p, Vector3& w_in, float& pdf )=0;
    virtual float pdf     ( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in )=0;
    virtual Color evaluate( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in )=0;
};


}

#endif // LEGION_INTERFACE_ISURFACESHADER_H_
