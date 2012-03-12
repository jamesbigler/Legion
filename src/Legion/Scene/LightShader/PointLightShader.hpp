
#ifndef LEGION_INTERFACE_POINTLIGHTSHADER_H_
#define LEGION_INTERFACE_POINTLIGHTSHADER_H_


#include <Legion/Scene/LightShader/ILightShader.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Core/Vector.hpp>
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
                    const LocalGeometry& p,
                    Vector3& on_light,
                    float& pdf )const;

    float   pdf( const LocalGeometry& p, const Vector3& w_in )const;


    Color   getPower()const;

    Color   getRadiance( const LocalGeometry& p,
                         const Vector3& w_in )const;

    //--------------------------------------------------------------------------
    // PointLightShader specific interface
    //--------------------------------------------------------------------------
    void setRadiantFlux( const Color& rflux );
    void setPosition( const Vector3& position );

private:
    Color   m_rflux;
    Vector3 m_position;

};


}

#endif // LEGION_INTERFACE_POINTLIGHTSHADER_H_
