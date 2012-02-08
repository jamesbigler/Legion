
/// \file ILightShader.hpp
/// Pure virtual interface for all LightShader classes
#ifndef LEGION_INTERFACE_ILIGHTSHADER_H_
#define LEGION_INTERFACE_ILIGHTSHADER_H_


#include <Private/APIBase.hpp>
#include <Core/Vector.hpp>


namespace legion
{

class Color;

/// Pure virtual interface for all LightShader classes
class ILightShader : public APIBase
{
public:

    /// Create a named ILightShader object
    ///   \param name  The object's name 
    explicit        ILightShader( const std::string& name );


    /// Destroy an ILightShader object
    virtual         ~ILightShader();

    /// Sample the solid angle subtended by the light from the given point 
    ///   \param  p     The local geometry of the surface being shaded
    ///   \param  w_in  The sampled direction to the light 
    ///   \param  pdf   The PDF of the given sample direction
    virtual void    sample( const SurfaceGeometry& p,
                            Vector3& w_in,
                            float& pdf )=0;

    /// Query the pdf of a given to-light direction.
    ///   \param  p     The local geometry of the surface being shaded
    ///   \param  w_in  The direction to the light 
    ///   \returns the pdf
    virtual float   pdf( const SurfaceGeometry& p, const Vector3& w_in )=0;


    /// Query the total power emitted by this light
    ///   \returns  The emitted power
    virtual float   getPower()const=0;

    /// Evaluate the incident radiance from the light towards the given surface 
    /// geometry.  Should perform no occlusion testing
    ///   \param p     The local geometry info of the surface being shaded
    ///   \param w_in  Direction towards the light
    ///   \returns The incident radiance
    virtual Color   getRadiance( const SurfaceGeometry& p,
                                 const Vector3& w_in )=0;
};


}

#endif // LEGION_INTERFACE_ILIGHTSHADER_H_
