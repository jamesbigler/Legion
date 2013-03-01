
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

#include <Legion/Objects/cuda_common.hpp>
#include <Legion/Objects/Surface/CUDA/Surface.hpp>
#include <Legion/Objects/Surface/CUDA/Microfacet.hpp>
#include <Legion/Objects/Texture/CUDA/Texture.hpp>
#include <Legion/Common/Math/CUDA/ONB.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>

rtDeclareVariable( float3, reflectance, , );


using namespace legion;

RT_CALLABLE_PROGRAM
legion::BSDFSample beckmannSampleBSDF( 
        float3 seed,
        float3 w_out,
        legion::LocalGeometry p )
{
    //float  absorption = ...
    // float eta = ....
    BeckmannDistribution distribution( 0.02 );
    NopFresnel           fresnel;

    MicrofacetSurface<BeckmannDistribution, NopFresnel> 
        surface( reflectance, distribution, fresnel );
    return surface.sample( make_float2( seed ), w_out, p );
}


RT_CALLABLE_PROGRAM
float4 beckmannEvaluateBSDF(
        float3 w_out,
        legion::LocalGeometry p,
        float3 w_in )
{
    BeckmannDistribution distribution( 0.02 );
    NopFresnel           fresnel;

    MicrofacetSurface<BeckmannDistribution, NopFresnel> 
        surface( reflectance, distribution, fresnel );
    return surface.evaluate( w_out, p, w_in );
}


RT_CALLABLE_PROGRAM
float beckmannPDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    BeckmannDistribution distribution( 0.02 );
    NopFresnel           fresnel;

    MicrofacetSurface<BeckmannDistribution, NopFresnel> 
        surface( reflectance, distribution, fresnel );
    return surface.pdf( w_out, p, w_in );
}
