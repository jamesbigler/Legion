
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

#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Noncopyable.hpp>
#include <Legion/Core/APIBase.hpp>
#include <Legion/Core/Declarations.hpp>
#include <Legion/Renderer/Renderer.hpp>

#include <iostream>

namespace legion
{

class ICamera;
class IFilm;
class ILightShader;
class Mesh;

class Context : public APIBase 
{
public:
    //
    // External interface -- will be wrapped via Pimpl
    //
    explicit   Context( const std::string& name );
    ~Context();

    void addMesh( Mesh* mesh );

    void addLight( const ILightShader* light_shader );

    void setActiveCamera( ICamera* camera );

    void setActiveFilm  ( IFilm* film );

    void setPixelFIlter( FilterType type );

    void setSamplesPerPixel( const Index2& spp  );
    void setMaxRayDepth( unsigned max_depth );

    void render();

    //
    // Internal interface -- will exist only in Pimpl class
    //
    optix::Buffer createOptixBuffer();


private:


    std::vector<const Mesh*> m_meshes;
    const ICamera*           m_camera;
    const IFilm*             m_film;

    Renderer                 m_renderer;
};


}
#endif // LEGION_CORE_CONTEXT_H_
