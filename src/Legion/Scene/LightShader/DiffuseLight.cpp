 
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


#include <Legion/Core/Color.hpp>
#include <Legion/Renderer/Cuda/Shared.hpp>
#include <Legion/Scene/LightShader/DiffuseLight.hpp>


using namespace legion;


DiffuseLight::DiffuseLight( Context* context, const std::string& name )
    : MeshLight( context, name ),
      m_emittance( 0.0f )
{
}

    
DiffuseLight::~DiffuseLight()
{
}

    
Color DiffuseLight::power()const
{
    LEGION_TODO();
}


Color DiffuseLight::emittance( const LocalGeometry& light_geom,
                            const Vector3& w_in )const
{
    return m_emittance * fmaxf( 0.0f,  dot( light_geom.shading_normal, -w_in ) );
}


void DiffuseLight::setEmittance( const Color& emittance )
{
    m_emittance = emittance;
}
