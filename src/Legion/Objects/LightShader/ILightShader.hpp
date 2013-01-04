
/// \file ILightShader.hpp
/// Pure virtual interface for all LightShader classes
#ifndef LEGION_INTERFACE_ILIGHTSHADER_HPP_
#define LEGION_INTERFACE_ILIGHTSHADER_HPP_


#include <Legion/Core/APIBase.hpp>
#include <Legion/Common/Math/Vector.hpp>


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

    virtual         isEmitter()const=0;

    virtual bool    isSingular()const=0;

    // light_p is the geometry of the light, w_in is the incoming direction 
    // to that point on the light
    virtual Color   emittance( const LocalGeometry& light_geom,
                               const Vector3& w_in );

};


}

#endif // LEGION_INTERFACE_ILIGHTSHADER_HPP_
