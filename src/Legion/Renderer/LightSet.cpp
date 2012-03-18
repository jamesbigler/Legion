 
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
#include <Legion/Scene/LightShader/ILightShader.hpp>

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
    m_light_vec.push_back( light );
    m_light_map.insert( std::make_pair( light->getID(), light ) );
}


void LightSet::removeLight( const ILightShader* light )
{
    LightVec::iterator it = std::remove( m_light_vec.begin(),
                                         m_light_vec.end(),
                                         light );
    m_light_vec.erase( it, m_light_vec.end() );

    m_light_map.erase( light->getID() );
}


size_t LightSet::numLights()const
{
    return m_light_vec.size();
}


void LightSet::selectLight( float rnd,
                            const ILightShader*& light,
                            float& pdf )const
{
    if( m_light_vec.empty() )
    {
        light = 0u;
        pdf   = 0.0f;
        return;
    }

    size_t   num_lights   = m_light_vec.size();
    float    num_lights_f = static_cast<float>( num_lights ); 
    unsigned idx = std::min( static_cast<unsigned>( rnd * num_lights_f ),
                             static_cast<unsigned>( num_lights-1 ) ); 
    light = m_light_vec[ idx ];
    pdf   = 1.0f / num_lights_f;
}


const ILightShader* LightSet::lookupLight( unsigned id )const
{
    LightMap::const_iterator it = m_light_map.find( id );
    return (it == m_light_map.end() )? 0u : it->second;
}

