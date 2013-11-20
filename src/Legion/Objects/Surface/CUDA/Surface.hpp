
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
#include <Legion/Objects/Light/CUDA/Light.hpp>


namespace legion
{

enum BSDFEventType
{
    BSDF_EVENT_SPECULAR=0,
    BSDF_EVENT_DIFFUSE
};

struct BSDFSample
{
    float3 w_in;
    float3 f_over_pdf;
    float pdf;
    short is_singular;
    short event_type;
};

}

rtDeclareVariable( float, legionSurfaceArea, , );

rtCallableProgram( legion::BSDFSample,
                   legionSurfaceSampleBSDF,
                   ( float3, float3 , legion::LocalGeometry ) );

// ( bsdf,pdf ) legionSurfaceEvaluateBSDF(
//                   float3 w_in,
//                   LocalGeometry p,
//                   float3 w_out)
rtCallableProgram( float4, 
                   legionSurfaceEvaluateBSDF,
                   ( float3 , legion::LocalGeometry, float3 ) );

// pdf          legionSurfacePDF(
//                   float3 w_in,
//                   LocalGeometry p,
//                   float3 w_out)
rtCallableProgram( float, 
                   legionSurfacePDF,
                   ( float3, legion::LocalGeometry, float3 ) );

// emittance    legionSurfaceEmission( point_on_light )
rtCallableProgram( float3, 
                   legionSurfaceEmission,
                   ( legion::LightSample ) ); 

//------------------------------------------------------------------------------
//
// Null surface programs
//
//------------------------------------------------------------------------------

RT_CALLABLE_PROGRAM
float3 nullSurfaceEmission( legion::LightSample )
{
    return make_float3( 0.0f );
}


RT_CALLABLE_PROGRAM
legion::BSDFSample nullSurfaceSampleBSDF( float3 seed, float3 w_out, legion::LocalGeometry p )
{
    legion::BSDFSample sample;
    sample.w_in        = make_float3( 0.0f );
    sample.f_over_pdf  = make_float3( 0.0f );
    sample.pdf         = 0.0f;
    sample.is_singular = 1;
    return sample;
}


RT_CALLABLE_PROGRAM
float4 nullSurfaceEvaluateBSDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    return make_float4( 0.0f ); 
}


RT_CALLABLE_PROGRAM
float nullSurfacePDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    return 0.0f;
}

//------------------------------------------------------------------------------
//
// Nested surfaces
//
//------------------------------------------------------------------------------

#define legionDeclareSurface( name )                                           \
    rtCallableProgram( float4,                                                 \
                       name ## _evaluateBSDF__,                                \
                       ( float3, legion::LocalGeometry, float3 ) );            \
    rtCallableProgram( float,                                                  \
                       name ## _PDF__,                                         \
                       ( float3, legion::LocalGeometry, float3 ) );            \
    rtCallableProgram( float3,                                                 \
                       name ## _emission__,                                    \
                       ( float3, legion::LocalGeometry ) );                    \
    rtCallableProgram( legion::BSDFSample,                                     \
                       name ## _sampleBSDF__,                                  \
                       ( float3, float3 , legion::LocalGeometry ) );

      
#define legionEvaluateBSDF( name, w_out, p, w_in )                             \
    name ## _evaluateBSDF__( w_out, p, w_in )

#define legionPDF( name, w_out, p, w_in )                                      \
    name ## _PDF__( w_out, p, w_in )

#define legionSampleBSDF( name, seed, w_out, p )                               \
    name ## _sampleBSDF__( seed, w_out, p )

#define legionEmission( name, w_in, on_light )                                 \
    name ## _emission__( w_in, on_light )


#endif // LEGION_OBJECTS_SURFACE_CUDA_SURFACE_HPP_
