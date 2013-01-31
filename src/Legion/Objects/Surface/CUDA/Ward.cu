
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
#include <Legion/Common/Math/CUDA/ONB.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>

rtDeclareVariable( float3, reflectance, , );

RT_CALLABLE_PROGRAM
legion::BSDFSample wardSampleBSDF( 
        float2 seed,
        float3 w_out,
        legion::LocalGeometry p )
{
    legion::BSDFSample sample;

    // sample hemisphere with cosine density by uniformly sampling
    // unit disk and projecting up to hemisphere
    float2 on_disk( legion::squareToDisk( seed ) );
    const float x = on_disk.x;
    const float y = on_disk.y;
          float z = 1.0f - x*x -y*y;

    z = z > 0.0f ? sqrtf( z ) : 0.0f;

    // Transform into world space
    legion::ONB onb( p.shading_normal );
    sample.w_in = onb.inverseTransform( make_float3( x, y, z ) );

    // calculate pdf
    sample.pdf        = z * legion::ONE_DIV_PI;
    sample.f_over_pdf = reflectance;

    return sample;
}


RT_CALLABLE_PROGRAM
float4 wardEvaluateBSDF(
        float3 w_out,
        legion::LocalGeometry p,
        float3 w_in )
{
    const float cosine = fmaxf( 0.0f, optix::dot( w_in, p.shading_normal ) );
    const float pdf    = cosine * legion::ONE_DIV_PI;
    return make_float4( pdf * reflectance, pdf );
}


RT_CALLABLE_PROGRAM
float wardPDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    float cosine = fmaxf( 0.0f, optix::dot( w_in, p.shading_normal ) );
    return cosine * legion::ONE_DIV_PI;
}
