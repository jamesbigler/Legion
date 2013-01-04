
#ifndef LEGION_INTERFACE_POINTLIGHTSHADER_H_
#define LEGION_INTERFACE_POINTLIGHTSHADER_H_


#include <Legion/Objects/LightShader/ILightShader.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <tr1/memory>


namespace legion
{

struct LocalGeometry;


class PointLightShader : public ILightShader 
{
public:
    PointLightShader( Context* context, const std::string& name );
    virtual ~PointLightShader();

    //--------------------------------------------------------------------------
    // ILightShader interface
    //--------------------------------------------------------------------------
    void    sample( const Vector2& seed,
                    const LocalGeometry& geom,
                    Vector3& on_light,
                    float& pdf )const;

    bool    isSingular()const;

    Color   power()const;

    Color   emittance( const LocalGeometry& light_geom,
                       const Vector3& w_in )const;

    //--------------------------------------------------------------------------
    // PointLightShader specific interface
    //--------------------------------------------------------------------------
    void setIntensity( const Color& intensity );
    void setPosition( const Vector3& position );

private:
    Color   m_intensity;
    Vector3 m_position;

};


}

#endif // LEGION_INTERFACE_POINTLIGHTSHADER_H_
