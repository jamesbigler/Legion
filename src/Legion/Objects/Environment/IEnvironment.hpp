
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


/// \file IEnvironment.hpp
/// Pure virtual interface for Environment classes


#ifndef LEGION_OBJECTS_ENVIRONMENT_IENVIRONMENT_HPP_
#define LEGION_OBJECTS_ENVIRONMENT_IENVIRONMENT_HPP_

#include <Legion/Objects/IObject.hpp>

namespace legion
{


/// Pure virtual interface for Environment objects
class IEnvironment : public IObject
{
public:
    IEnvironment( Context* context ) : IObject( context ) {}

    virtual ~IEnvironment() {}

    /// Return the name of this Environment type.  The associated PTX file should
    /// be named {name()}.ptx
    virtual const char* name()const=0;

    virtual const char* missEvaluateFunctionName()const=0;

    virtual const char* lightEvaluateFunctionName()const=0;

    virtual const char* sampleFunctionName()const=0;
};


}

#endif // LEGION_OBJECTS_ENVIRONMENT_IENVIRONMENT_HPP_
