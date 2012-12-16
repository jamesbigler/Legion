

/// \file ISurfaceShader.hpp
/// Pure virtual interface for Surface Shaders
#ifndef LEGION_SCENE_SURFACESHADER_ISURFACESHADER_H_
#define LEGION_SCENE_SURFACESHADER_ISURFACESHADER_H_





namespace legion
{


class VariableContainer;

/// Pure virtual interface for Surface Shaders
class ISurfaceShader 
{
public:
    virtual ~ISurfaceShader() {}
    
    /// Return the name of this SurfaceShader type
    /// static std::string name();

    /// Return the name of this Camera's cuda ray generation function 
    /// static std::string rayGenFunctionName();

    /// Set variables needed by the ray generation function 
    virtual void setVariables( VariableContainer& container ) const = 0;



    /*
    /// Destroy ISurfaceShader object
    virtual         ~ISurfaceShader();

    /// Sample the BSDF at given local geometry.
    ///   \param      seed   2D sampling seed in [0,1]^2
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param[out] w_in   Direction to light
    ///   \param[out] pdf    The BSDF PDF for w_in 
    virtual void    sampleBSDF( const Vector2& seed,
                                const Vector3& w_out,
                                const LocalGeometry& p,
                                Vector3& w_in,
                                Color& f_over_pdf )const=0;

    virtual bool    isSingular()const=0;


    /// Compute the PDF value for the given incoming/outgoing direction pair
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param      w_in   Direction to light
    ///   \returns The pdf
    virtual float   pdf( const Vector3& w_out,
                         const LocalGeometry& p,
                         const Vector3& w_in )const=0;


    /// Compute the BSDF value for the given incoming/outgoing direction pair
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param      w_in   Direction to light
    ///   \returns The value of the bsdf
    virtual Color   evaluateBSDF( const Vector3& w_out,
                                  const LocalGeometry& p,
                                  const Vector3& w_in )const=0;
    */
};


}

#endif // LEGION_SCENE_SURFACESHADER_ISURFACESHADER_H_
