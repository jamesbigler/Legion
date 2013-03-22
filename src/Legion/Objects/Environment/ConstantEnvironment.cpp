
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

#include <Legion/Objects/Environment/ConstantEnvironment.hpp>
#include <Legion/Objects/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>


using namespace legion;

IEnvironment* ConstantEnvironment::create(
        Context* context,
        const Parameters& params
        )
{
    ConstantEnvironment* const_env = new ConstantEnvironment( context );
    const_env->setRadiance( params.get( "radiance", Color(0.5f, 0.5f, 0.5f) ) );
    return const_env;
}


ConstantEnvironment::ConstantEnvironment( Context* context )
    : IEnvironment( context ),
      m_radiance( 0.5f, 0.5f, 0.5f )
{
}


ConstantEnvironment::~ConstantEnvironment()
{
}


const char* ConstantEnvironment::name() const
{
    return "ConstantEnvironment";
}


const char* ConstantEnvironment::missEvaluateFunctionName() const
{
    return "constantEnvironmentMissEvaluate";
}


const char* ConstantEnvironment::lightEvaluateFunctionName() const
{
    return "constantEnvironmentLightEvaluate";
}


const char* ConstantEnvironment::sampleFunctionName() const
{
    return "constantEnvironmentSample";
}


void ConstantEnvironment::setRadiance( const Color& radiance )
{
    m_radiance = radiance;
}


void ConstantEnvironment::setVariables( VariableContainer& container ) const
{
    container.setFloat( "radiance", m_radiance );
}
