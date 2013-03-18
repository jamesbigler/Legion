
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
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

#include <Legion/Common/Util/Factory.hpp>
#include <Legion/Common/Util/Assert.hpp>

using namespace legion;

#define DEFINE_FACTORY_FUNCTIONS( plugin, plugin_map )                         \
    void Factory::registerObject(                                              \
            const std::string& name,                                           \
            plugin ## Creator creator )                                        \
    {                                                                          \
        LEGION_ASSERT( creator != 0 );                                         \
        if( plugin_map.count( name ) != 0 )                                    \
            throw Exception( "Plugin creator '" + name + "' already "          \
                             "registered" );                                   \
        plugin_map.insert( std::make_pair( name, creator ) );                  \
    }                                                                          \
                                                                               \
    I ## plugin* Factory::create ## plugin(                                    \
            const std::string& name,                                           \
            const Parameters& p )                                              \
    {                                                                          \
        plugin ## Creators::iterator it = plugin_map.find( name );             \
        if( it == plugin_map.end() )                                           \
            throw Exception( "Unknown plugin creator '" + name + "'" );        \
        return it->second( m_context, p );                                     \
    }


DEFINE_FACTORY_FUNCTIONS( Camera,      m_camera_creators      );
DEFINE_FACTORY_FUNCTIONS( Display,     m_display_creators     );
DEFINE_FACTORY_FUNCTIONS( Environment, m_environment_creators );
DEFINE_FACTORY_FUNCTIONS( Geometry,    m_geometry_creators    );
DEFINE_FACTORY_FUNCTIONS( Light,       m_light_creators       );
DEFINE_FACTORY_FUNCTIONS( Surface,     m_surface_creators     );
DEFINE_FACTORY_FUNCTIONS( Renderer,    m_renderer_creators    );
DEFINE_FACTORY_FUNCTIONS( Texture,     m_texture_creators     );
