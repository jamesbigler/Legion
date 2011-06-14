
#ifndef LEGION_INTERFACE_ISHADER_H_
#define LEGION_INTERFACE_ISHADER_H_


#include <Private/APIBase.hpp>
#include <Core/Color.hpp>
#include <Core/Vector.hpp>


namespace legion
{

namespace Shader 
{
    class Geometry
    {
    };
}

class IShader : public APIBase
{
public:
    IShader( const std::string& name );
    virtual ~IShader();

    virtual void  sample  ( const Vector3& w_out, const Shader::Geometry& p, Vector3& w_in, float& pdf )=0;
    virtual float pdf     ( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in )=0;
    virtual Color evaluate( const Vector3& w_out, const Shader::Geometry& p, const Vector3& w_in )=0;
};


}

#endif // LEGION_INTERFACE_ISHADER_H_
