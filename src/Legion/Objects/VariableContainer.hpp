
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


#ifndef LEGION_CORE_VARIABLE_CONTAINER_H_
#define LEGION_CORE_VARIABLE_CONTAINER_H_

#include <Legion/Common/Math/Vector.hpp>
#include <optixu/optixpp_namespace.h>
#include <vector>

namespace legion
{

class Matrix;
class Color;
class ISurface;
class ITexture;

class VariableContainer
{
public:
    typedef std::vector< std::pair<std::string, const ITexture*> > Textures;
    typedef std::vector< std::pair<std::string, const ISurface*> > Surfaces;

    explicit VariableContainer( optix::ScopedObj* scoped );

    void setFloat   ( const std::string& name, const Vector2& val )const;
    void setFloat   ( const std::string& name, const Vector3& val )const;
    void setFloat   ( const std::string& name, const Vector4& val )const;
    void setFloat   ( const std::string& name, const Color& val   )const;
    void setFloat   ( const std::string& name, float val          )const;
    void setUnsigned( const std::string& name, unsigned val       )const;
    void setMatrix  ( const std::string& name, const Matrix& val  )const;
    void setBuffer  ( const std::string& name, optix::Buffer val  )const;
    void setTexture ( const std::string& name, const ITexture* val);
    void setSurface ( const std::string& name, const ISurface* val);

    
    const Textures& getTextures()const   { return m_textures; }
    const Surfaces& getSurfaces()const   { return m_surfaces; }
private:
    optix::ScopedObj*  m_scoped;
    Textures           m_textures;
    Surfaces           m_surfaces;
};

}


#endif // LEGION_CORE_VARIABLE_CONTAINER_H_
