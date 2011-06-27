
#ifndef LEGION_INTERFACE_ILIGHTSHADER_H_
#define LEGION_INTERFACE_ILIGHTSHADER_H_


#include <Private/APIBase.hpp>
#include <Core/Color.hpp>
#include <Core/Color.hpp>
#include <Core/Vector.hpp>


namespace legion
{

    namespace Shader
    {
        class Geometry;
    }


    class ILightShader : public APIBase
    {
    public:
        ILightShader( const std::string& name );
        virtual ~ILightShader();

        virtual void  sample  ( const Shader::Geometry& p, Vector3& w_in, float& pdf )=0;
        virtual float pdf     ( const Shader::Geometry& p, const Vector3& w_in )=0;
        virtual Color evaluate( const Shader::Geometry& p, const Vector3& w_in )=0;
    };


}

#endif // LEGION_INTERFACE_ILIGHTSHADER_H_
