
// Copyright (C) 2011 R. Keith Morley
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// (MIT/X11 License)

/// \file MeshLight.hpp
/// MeshLight

#ifndef LEGION_SCENE_LIGHTSHADER_MESHLIGHT_HPP_
#define LEGION_SCENE_LIGHTSHADER_MESHLIGHT_HPP_

#include <Legion/Scene/LightShader/ILightShader.hpp>


// TODO: change name to IMeshLightShader and same for IBasicCamera
namespace legion
{

class Context;
class Mesh;
struct LocalGeometry;

class MeshLight : public  ILightShader
{
public:
    MeshLight( Context* context, const std::string& name );

    //--------------------------------------------------------------------------
    // ILightShader interface
    //--------------------------------------------------------------------------

    virtual ~MeshLight();
    
    /// Samples the associated mesh
    void    sample( const Vector2& seed,
                    const LocalGeometry& lgeom,
                    Vector3& on_light,
                    float& pdf )const;

    /// Returns true
    bool    isSingular()const;

    virtual Color power()const=0;

    virtual Color emittance( const LocalGeometry& light_geom,
                             const Vector3& w_in )const=0;


    //--------------------------------------------------------------------------
    // INTERNAL interface.  PIMPLed in future
    //--------------------------------------------------------------------------
    void setMesh( Mesh* mesh );

private:
    Mesh*   m_mesh; // TODO: PIMPL
};


}


#endif // LEGION_SCENE_LIGHTSHADER_MESHLIGHT_HPP_
