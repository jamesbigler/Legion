
#ifndef LEGION_INTERFACE_POINTLIGHTSHADER_H_
#define LEGION_INTERFACE_POINTLIGHTSHADER_H_


#include <Legion/Scene/LightShader/ILightShader.hpp>
#include <Legion/Core/Vector.hpp>
#include <tr1/memory>


namespace legion
{

class LocalGeometry;
class Color;


class PointLightShader : public ILightShader 
{
public:
    PointLightShader( Context* context, const std::string& name );
    virtual ~PointLightShader();

    //--------------------------------------------------------------------------
    // ILightShader interface
    //--------------------------------------------------------------------------
    void    sample( const LocalGeometry& p,
                            Vector3& w_in,
                            float& pdf );

    float   pdf( const LocalGeometry& p, const Vector3& w_in );


    float   getPower()const;

    Color   getRadiance( const LocalGeometry& p,
                         const Vector3& w_in );

    //--------------------------------------------------------------------------
    // PointLightShader specific interface
    //--------------------------------------------------------------------------
    void setRadiantFlux( const Color& kd );
    void setPosition( const Vector3& position );

private:
    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;

};


}

#endif // LEGION_INTERFACE_POINTLIGHTSHADER_H_
