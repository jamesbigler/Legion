
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


#ifndef LEGION_OBJECTS_GEOMETRY_IGEOMETRY_H_
#define LEGION_OBJECTS_GEOMETRY_IGEOMETRY_H_

#include <Legion/Objects/IObject.hpp>
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>

namespace legion
{

class Matrix;
class ISurface;

/// Pure virtual interface for Geometry objects
class LCLASSAPI IGeometry : public IObject
{
public:
    LAPI IGeometry( Context* context ) : IObject( context ) {}
    LAPI virtual ~IGeometry() {}

    LAPI virtual const char* name()                    const=0;
    LAPI virtual const char* intersectionFunctionName()const=0;
    LAPI virtual const char* boundingBoxFunctionName() const=0;
    LAPI virtual const char* sampleFunctionName()      const=0;
    LAPI virtual const char* pdfFunctionName()         const=0;

    LAPI virtual unsigned    numPrimitives()const=0;
    LAPI virtual float       area()const=0;

    LAPI virtual void        setTransform( const Matrix& transform )=0;
    LAPI virtual Matrix      getTransform() const=0;

    LAPI virtual void        setSurface( ISurface* surface )=0;
    LAPI virtual ISurface*   getSurface()const = 0 ;
};



class LCLASSAPI Instance : public IGeometry
{
public:
    LAPI Instance( Context* context, IGeometry* child, const Matrix& transform )
        : IGeometry( context ),
          m_transform( transform ), 
          m_child( child ),
          m_surface( 0 )
    {}

    LAPI ~Instance() {}

    LAPI const char* name()const
    { return m_child->name(); }

    LAPI const char* intersectionFunctionName()const
    { return m_child->intersectionFunctionName(); }

    LAPI const char* boundingBoxFunctionName()const
    { return m_child->boundingBoxFunctionName(); }

    LAPI const char* sampleFunctionName()const
    { return m_child->sampleFunctionName(); }

    LAPI virtual unsigned numPrimitives()const
    { return m_child->numPrimitives(); }

    LAPI void setTransform( const Matrix& transform )
    { m_transform = transform; }

    LAPI Matrix getTransform() const
    { return m_transform * m_child->getTransform(); }

    LAPI void setSurface( ISurface* surface )
    { m_surface = surface; }
    
    LAPI ISurface* getSurface()const 
    { return m_surface ? m_surface : m_child->getSurface(); }

private:
    Matrix     m_transform;
    IGeometry* m_child;
    ISurface*  m_surface;
};

}

#endif // LEGION_OBJECTS_GEOMETRY_IGEOMETRY_H_
