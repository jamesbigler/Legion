 

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

#include <Legion/Renderer/LightSet.hpp>

#include <algorithm>
#include <cmath>

using namespace legion;


LightSet::LightSet()
{
}


LightSet::~LightSet()
{
}


void LightSet::addLight( const ILightShader* light )
{
    m_lights.push_back( light );
}


void LightSet::removeLight( const ILightShader* light )
{
    Lights::iterator it = std::remove( m_lights.begin(),
                                       m_lights.end(),
                                       light );
    m_lights.erase( it, m_lights.end() );
}


size_t LightSet::numLights()const
{
    return m_lights.size();
}


void LightSet::selectLight( float rnd,
                            const ILightShader*& light,
                            float& pdf )const
{
    if( m_lights.empty() )
    {
        light = 0u;
        pdf   = 0.0f;
        return;
    }

    size_t   num_lights   = m_lights.size();
    float    num_lights_f = static_cast<float>( num_lights ); 
    unsigned idx = std::min( static_cast<unsigned>( rnd * num_lights_f ),
                             static_cast<unsigned>( num_lights-1 ) ); 
    light = m_lights[ idx ];
    pdf   = 1.0f / num_lights_f;
}

