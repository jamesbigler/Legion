
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

/// \file ISurface.hpp
/// Pure virtual interface for Surface Shaders
#ifndef LEGION_OBJECTS_SURFACE_ISURFACE_HPP_
#define LEGION_OBJECTS_SURFACE_ISURFACE_HPP_

#include <Legion/Objects/IObject.hpp>


namespace legion
{

/// Pure virtual interface for Surfaces
class ISurface : public IObject
{
public:
    ISurface( Context* context ) : IObject( context ) {}

    virtual ~ISurface() {}


    virtual const char* name()const=0;
    virtual const char* sampleBSDFFunctionName()const=0;
    virtual const char* evaluateBSDFFunctionName()const=0;
    // TODO: make this bsdfPDFFunctionName
    virtual const char* pdfFunctionName()const=0;
    virtual const char* emissionFunctionName()const=0;

};


}

#endif // LEGION_OBJECTS_SURFACE_ISURFACE_HPP_
