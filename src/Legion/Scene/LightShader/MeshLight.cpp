
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

#include <Legion/Common/Util/Assert.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Scene/LightShader/MeshLight.hpp>
#include <Legion/Scene/Mesh/Mesh.hpp>

using namespace legion;


MeshLight::MeshLight( Context* context,
                      const std::string& name,
                      Mesh* mesh,
                      /*const AreaLight* */ const ILightShader* light_shader )
  : ILightShader( context, name ),
    m_mesh( mesh ),
    m_light_shader( light_shader )

{
}


MeshLight::~MeshLight()
{
}


void MeshLight::sample( const Vector2& seed,
                               Vector3& on_light,
                               float& pdf )const
{
    m_mesh->sample( seed, on_light, pdf );
}


bool MeshLight::isSingular()const
{
    return false;
}


Color MeshLight::power()const
{
    LEGION_TODO();
    
    return Color( 0.0f );
}


Color MeshLight::emittance( const LocalGeometry& light_geom,
                                   const Vector3& w_in )const
{
    return m_light_shader->emittance( light_geom, w_in );
    //return Color( 1000.0f ); 
}
    
