

/// \file ISurfaceShader.hpp
/// Pure virtual interface for Surface Shaders
#ifndef LEGION_INTERFACE_ISURFACESHADER_H_
#define LEGION_INTERFACE_ISURFACESHADER_H_


#include <Private/APIBase.hpp>
#include <Core/Vector.hpp>



namespace legion
{

class Color;

/// Pure virtual interface for Surface Shaders
class ISurfaceShader : public APIBase
{
public:
    /// Create named ISurfaceShader object
    explicit       ISurfaceShader( const std::string& name );
    
    /// Destroy ISurfaceShader object
    virtual        ~ISurfaceShader();

    /// Sample the BSDF at given local geometry.
    ///   \param      seed   2D sampling seed in [0,1]^2
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param[out] w_in   Direction to light
    ///   \param[out] pdf    The BSDF PDF for w_in 
    virtual void   sampleBSDF( const Vector2& seed,
                               const Vector3& w_out,
                               const SurfaceGeometry& p,
                               Vector3& w_in,
                               float& pdf )=0;


    /// Compute the PDF value for the given incoming/outgoing direction pair
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param      w_in   Direction to light
    ///   \returns The pdf
    virtual float   pdf( const Vector3& w_out,
                         const SurfaceGeometry& p,
                         const Vector3& w_in )=0;

    /// Compute the BSDF value for the given incoming/outgoing direction pair
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param      w_in   Direction to light
    ///   \returns The value of the bsdf
    virtual Color   evalueateBSDF( const Vector3& w_out,
                                   const SurfaceGeometry& p,
                                   const Vector3& w_in )=0;
};


}

#endif // LEGION_INTERFACE_ISURFACESHADER_H_
