
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

#include <Legion/Objects/Surface/Mixture.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/Parameters.hpp>

using namespace legion;
  

ISurface* Mixture::create( Context* context, const Parameters& params )
{
    Mixture* mixture = new Mixture( context );
    ISurface* s0  = params.get( "s0", static_cast<ISurface*>( 0 ) );
    ISurface* s1  = params.get( "s1", static_cast<ISurface*>( 0 ) );
    mixture->setSurfaces( s0, s1 );

    mixture->setMixtureWeight( 
            params.get( "mixture_weight", static_cast<ITexture*>( 0 ) )
            );

    return mixture;
}


Mixture::Mixture( Context* context )
  : ISurface( context ),
    m_s0( 0 ),
    m_s1( 0 ),
    m_mixture_weight( 0 )
{
}


Mixture::~Mixture()
{
}


void Mixture::setSurfaces( const ISurface* s0, const ISurface* s1 )
{
    LEGION_ASSERT( s0 );
    LEGION_ASSERT( s1 );
    m_s0 = s0;
    m_s1 = s1;
}


void Mixture::setMixtureWeight( const ITexture* mixture_weight )
{
    LEGION_ASSERT( mixture_weight );
    m_mixture_weight = mixture_weight;
}


const char* Mixture::name()const
{
    return "Mixture";
}


const char* Mixture::sampleBSDFFunctionName()const
{
    return "mixtureSampleBSDF";
}


const char* Mixture::evaluateBSDFFunctionName()const
{
    return "mixtureEvaluateBSDF";
}


const char* Mixture::pdfFunctionName()const
{
    return "mixturePDF";
}
    

const char* Mixture::emissionFunctionName()const
{
    return "nullSurfaceEmission";
}


void Mixture::setVariables( VariableContainer& container ) const
{
    LEGION_ASSERT( m_s0             );
    LEGION_ASSERT( m_s1             );
    LEGION_ASSERT( m_mixture_weight );

    container.setSurface( "surf0", m_s0 );
    container.setSurface( "surf1", m_s1 );
    container.setTexture( "mixture_weight", m_mixture_weight );
}
