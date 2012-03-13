
/// \file ILightShader.hpp
/// Pure virtual interface for all LightShader classes
#ifndef LEGION_INTERFACE_ILIGHTSHADER_HPP_
#define LEGION_INTERFACE_ILIGHTSHADER_HPP_


#include <Legion/Core/APIBase.hpp>
#include <Legion/Core/Vector.hpp>


namespace legion
{

class  Color;
struct LocalGeometry;

/// Pure virtual interface for all LightShader classes
class ILightShader : public APIBase
{
public:

    /// Create a named ILightShader object
    ///   \param name  The object's name 
                    ILightShader( Context* context, const std::string& name );

    /// Destroy an ILightShader object
    virtual         ~ILightShader();

    /// Sample the light source based on area.  The renderer will convert this
    /// to a solid-angle based sample and pdf internally
    ///   \param  seed       Sampling seed 
    ///   \param  on_light   Sampled position on light 
    ///   \param  pdf        PDF of the given sample direction
    virtual void    sample( const Vector2& seed,
                            Vector3&       on_light,
                            float&         pdf )const=0;


    virtual bool    isSingular()const=0;

    /// Query the total power emitted by this light
    ///   \returns  The emitted power
    virtual Color   power()const=0;

    // light_p is the geometry of the light, w_in is the incoming direction 
    // to that point on the light
    virtual Color   emittance( const LocalGeometry& light_geom,
                               const Vector3& w_in )const=0;
};


}

#endif // LEGION_INTERFACE_ILIGHTSHADER_HPP_
