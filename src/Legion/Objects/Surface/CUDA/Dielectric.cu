
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
#include <Legion/Common/Math/CUDA/Math.hpp>


rtDeclareVariable( float, ior_in    , , );
rtDeclareVariable( float, ior_out   , , );
rtDeclareVariable( float, absorption, , );


RT_CALLABLE_PROGRAM
legion::BSDFSample dielectricSampleBSDF( 
        float3 seed,
        float3 w_out,
        legion::LocalGeometry p )
{
    legion::BSDFSample sample;

    sample.w_in        = optix::reflect( -w_out, p.shading_normal ); 
    sample.pdf         = 1.0f; 
    sample.f_over_pdf  = make_float3( 1.0f );
    sample.is_singular = true;

    return sample;
}


RT_CALLABLE_PROGRAM
float4 dielectricEvaluateBSDF(
        float3 w_out,
        legion::LocalGeometry p,
        float3 w_in )
{
    return make_float4( 0.0f ); 
}


RT_CALLABLE_PROGRAM
float dielectricPDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    return 0.0f;
}
