
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

#ifndef LEGION_OBJECTS_SURFACE_CUDA_SURFACE_HPP_
#define LEGION_OBJECTS_SURFACE_CUDA_SURFACE_HPP_

#include <Legion/Objects/cuda_common.hpp>


namespace legion
{
struct BSDFSample
{
    float3 w_in;
    float3 f_over_pdf;
    float pdf;
};

}

rtCallableProgram( legion::BSDFSample,
                   legionSurfaceSampleBSDF,
                   ( float2, float3 , legion::LocalGeometry ) );

// ( bsdf,pdf ) legionSurfaceEvaluateBSDF(
//                   float3 w_in,
//                   LocalGeometry p,
//                   float3 w_out)
rtCallableProgram( float4, 
                   legionSurfaceEvaluateBSDF,
                   ( float3 , legion::LocalGeometry, float3 ) );

rtCallableProgram( float, 
                   legionSurfacePDF,
                   ( float3, legion::LocalGeometry, float3 ) );

// (w_out, shading_point)
rtDeclareVariable( float, legionSurfaceArea, , );
rtCallableProgram( float3, 
                   legionSurfaceEmission,
                   ( float3, legion::LocalGeometry ) ); 

//------------------------------------------------------------------------------
//
// Null surface programs
//
//------------------------------------------------------------------------------

RT_CALLABLE_PROGRAM
float3 nullSurfaceEmission( float3 w_in, float3 light_point, float3 light_emission )
{
    return make_float3( 0.0f );
}


RT_CALLABLE_PROGRAM
legion::BSDFSample nullSurfaceSampleBSDF( float2 seed, float3 w_out, legion::LocalGeometry p )
{
    legion::BSDFSample sample;
    sample.w_in       = make_float3( 0.0f );
    sample.f_over_pdf = make_float3( 0.0f );
    return sample;
}


RT_CALLABLE_PROGRAM
float3 nullSurfaceEvaluateBSDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    return make_float3( 0.0f ); 
}


RT_CALLABLE_PROGRAM
float nullSurfacePDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    return 0.0f;
}

#endif // LEGION_OBJECTS_SURFACE_CUDA_SURFACE_HPP_
