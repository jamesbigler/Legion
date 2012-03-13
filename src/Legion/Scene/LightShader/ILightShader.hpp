
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

    /// Sample the solid angle subtended by the light from the given point 
    ///   \param  seed       Sampling seed 
    ///   \param  p          Local geometry of the surface being shaded
    ///   \param  on_light   Sampled position on light 
    ///   \param  pdf        PDF of the given sample direction
    virtual void    sample( const Vector2&       seed,
                            const LocalGeometry& p,
                            Vector3& on_light,
                            float&   pdf )const=0;

    virtual void    isSingular()const=0;

    /// Query the pdf of a given to-light direction.
    ///   \param  p     The local geometry of the surface being shaded
    ///   \param  w_in  The direction to the light 
    ///   \returns the pdf
    virtual float   pdf( const LocalGeometry& p, const Vector3& w_in )const=0;

    /// Query the total power emitted by this light
    ///   \returns  The emitted power
    virtual Color   power()const=0;

    /// TODO: this should evaluate the exitant radiance with regard to the surface info
    /// Evaluate the exitant radiance from the light towards the given surface 
    /// geometry.  Should perform no occlusion testing
    ///   \param p     The geometry info of a point on the light
    ///   \param w_in  Direction towards the light
    ///   \returns The exitant radiance
    virtual Color   radiantExitance( const LocalGeometry& p,
                                     const Vector3& w_in )const=0;
};


}

#endif // LEGION_INTERFACE_ILIGHTSHADER_HPP_
