
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


#ifndef LEGION_OBJECTS_ENVIRONMENT_CONSTANT_ENVIRONMENT_HPP_
#define LEGION_OBJECTS_ENVIRONMENT_CONSTANT_ENVIRONMENT_HPP_

#include <Legion/Objects/Environment/IEnvironment.hpp>
#include <Legion/Core/Color.hpp>

namespace legion
{


/// Pure virtual interface for Environment objects
class ConstantEnvironment : public IEnvironment 
{
public:
    ConstantEnvironment( Context* context );

    ~ConstantEnvironment();

    /// Return the name of this Environment type.  The associated PTX file
    /// should be named {name()}.ptx
    const char* name() const;


    const char* missEvaluateFunctionName()const;

    const char* lightEvaluateFunctionName()const;

    const char* sampleFunctionName() const;

    void setRadiance( const Color& radiance );
    
    void setVariables( const VariableContainer& container ) const;
private:

    Color m_radiance;
};


}

#endif // LEGION_OBJECTS_ENVIRONMENT_CONSTANT_ENVIRONMENT_HPP_
