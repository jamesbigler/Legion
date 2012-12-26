
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

/// \file IGeometry.hpp
/// Pure virtual interface for Geometry classes


#ifndef LEGION_SCENE_GEOMETRY_IGEOMETRY_H_
#define LEGION_SCENE_GEOMETRY_IGEOMETRY_H_

#include <Legion/Scene/ISceneObject.hpp>

namespace legion
{

class Matrix;
class ISurface;

/// Pure virtual interface for Geometry objects
class IGeometry : public ISceneObject
{
public:
    virtual ~IGeometry() {}

    virtual const char* getIntersectionName()const=0;
    virtual const char* getBoundingBoxName()const=0;

    virtual void        setTransform( const Matrix& transform )=0;
    virtual Matrix      getTransform() const=0;

    virtual void        setSurface( ISurface* surface )=0;
    virtual ISurface*   getSurface()const = 0 ;
};

}


#include <Legion/Common/Math/Matrix.hpp>

namespace legion
{

class ISurface;

class Instance : public IGeometry
{
public:
    Instance( IGeometry* child, const Matrix& transform )
        : m_transform( transform ), 
          m_child( child ),
          m_surface( 0 )
    {}

    ~Instance() {}

    const char* getIntersectionName()const
    { return m_child->getIntersectionName(); }

    const char* getBoundingBoxName()const
    { return m_child->getBoundingBoxName(); }

    void setTransform( const Matrix& transform )
    { m_transform = transform; }

    Matrix getTransform() const
    { return m_transform * m_child->getTransform(); }

    void setSurface( ISurface* surface )
    { m_surface = surface; }
    
    ISurface* getSurface()const 
    { return m_surface ? m_surface : m_child->getSurface(); }

private:
    Matrix     m_transform;
    IGeometry* m_child;
    ISurface*  m_surface;
};

}

#endif // LEGION_SCENE_GEOMETRY_IGEOMETRY_H_
