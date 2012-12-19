
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

#ifndef LEGION_COMMON_UTIL_PLUGIN_H_
#define LEGION_COMMON_UTIL_PLUGIN_H_

#include <string>
#include <memory>
#include <vector>

namespace legion
{

class ICamera;
class IFilm;
class IGeometry;
class ILight;
class ISurface;
class Parameters;


class Plugin
{
    

};

class PluginManager
{
public:
    PluginManager();
    ~PluginManager();

    std::vector<std::string> registeredPluginNames()const;

    template <typename PType>
    void registerPlugin( const std::string& name, 
                         PType* (*create)( const Parameters& params ) );

    template <typename PType>
    PType* create( const std::string& name, const Parameters& params );
private:
    class Impl;
    std::unique_ptr<Impl> m_impl;

};

#define FORWARD_DECLARE_PLUGIN_SPECIALIZATIONS( PLUGIN_TYPE )                  \
    template<>                                                                 \
    void PluginManager::registerPlugin< PLUGIN_TYPE >(                         \
            const std::string& plugin_name,                                    \
            PLUGIN_TYPE* (*create)( const Parameters& params ) );              \
                                                                               \
    template <>                                                                \
    PLUGIN_TYPE* PluginManager::create<PLUGIN_TYPE>(                           \
            const std::string& plugin_name,                                    \
            const Parameters& params );

FORWARD_DECLARE_PLUGIN_SPECIALIZATIONS( ICamera        )
FORWARD_DECLARE_PLUGIN_SPECIALIZATIONS( IFilm          )
FORWARD_DECLARE_PLUGIN_SPECIALIZATIONS( IGeometry      )
FORWARD_DECLARE_PLUGIN_SPECIALIZATIONS( ILight         )
FORWARD_DECLARE_PLUGIN_SPECIALIZATIONS( ISurface )

#undef FORWARD_DECLARE_PLUGIN_SPECIALIZATIONS

}

#endif // LEGION_COMMON_UTIL_PLUGIN_H_
