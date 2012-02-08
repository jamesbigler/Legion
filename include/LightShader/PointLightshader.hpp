
#ifndef LEGION_INTERFACE_POINTLIGHTSHADER_H_
#define LEGION_INTERFACE_POINTLIGHTSHADER_H_


#include <Interface/ILightShader.hpp>
#include <Core/Vector.hpp>
#include <tr1/memory>


namespace legion
{

class LocalGeometry;
class Color;


class PointLightShader : public ILightShader 
{
public:
    PointLightShader( const std::string& name );
    virtual ~PointLightShader();

    void  sample  ( const LocalGeometry& p, Vector3& w_in, float& pdf );
    float pdf     ( const LocalGeometry& p, const Vector3& w_in );
    Color evaluate( const LocalGeometry& p, const Vector3& w_in );

    void setRadiantFlux( const Color& kd );
    void setPosition( const Vector3& position );

private:
    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;

};


}

#endif // LEGION_INTERFACE_POINTLIGHTSHADER_H_
