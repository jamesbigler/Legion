
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

/// \file Parallelogram.hpp


#ifndef LEGION_OBJECTS_GEOMETRY_PARALLELOGRAM_HPP_
#define LEGION_OBJECTS_GEOMETRY_PARALLELOGRAM_HPP_

#include <Legion/Objects/Geometry/IGeometry.hpp>
#include <Legion/Common/Math/Vector.hpp>

namespace legion
{

class VariableContainer;
class Parameters;
class ISurface;

class Parallelogram : public IGeometry
{
public:
    static IGeometry* create( Context* context, const Parameters& params );

    Parallelogram( Context* context );
    Parallelogram( Context* context, const Parameters& params );
    
    const char* name()const;
    const char* intersectionFunctionName()const;
    const char* boundingBoxFunctionName()const;
    const char* sampleFunctionName()const;

    unsigned    numPrimitives()const;

    void        setTransform( const Matrix& transform );
    Matrix      getTransform() const;

    void        setSurface( ISurface* surface );
    ISurface*   getSurface()const;

    void        setAnchorUV( const Vector3& anchor,
                             const Vector3& U,
                             const Vector3& V );
                         
    void setVariables( const VariableContainer& container )const;

private:
    Matrix    m_transform;
    Vector3   m_anchor;
    Vector3   m_U;
    Vector3   m_V;
    ISurface* m_surface;
};


}


#endif // LEGION_OBJECTS_GEOMETRY_PARALLELOGRAM_HPP_
