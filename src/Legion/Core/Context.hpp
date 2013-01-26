
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

#ifndef LEGION_CORE_CONTEXT_H_
#define LEGION_CORE_CONTEXT_H_

#include <Legion/Common/Util/Noncopyable.hpp>
#include <Legion/Common/Math/Vector.hpp>

#include <memory>

namespace legion
{

class ICamera;
class IGeometry;
class ILight;
class IRenderer;
class ISurface;
class PluginContext;


class Context : public Noncopyable
{
public:
    class Impl;

    Context();
    ~Context();

    void setRenderer   ( IRenderer* renderer );
    void setCamera     ( ICamera* camera );
    void addGeometry   ( IGeometry* geometry );
    void addLight      ( ILight* light );


    void addAssetPath( const std::string& path );

    void render();

    PluginContext& getPluginContext();

private:
    //std::unique_ptr<Impl> m_impl;
    std::auto_ptr<Impl> m_impl;
};


}
#endif // LEGION_CORE_CONTEXT_H_
