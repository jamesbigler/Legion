
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

#ifndef LEGION_OBJECTS_SURFACE_CUDA_LIGHT_HPP_
#define LEGION_OBJECTS_SURFACE_CUDA_LIGHT_HPP_

#include <Legion/Objects/cuda_common.hpp>

namespace legion
{

struct LightSample 
{
    float3        w_in;        //< Direction to light
    float         distance;    //< Distance to light ( INFINITE if infinite light )
    float3        normal;      //< Normal on light ( -w_in if singular light )
    float         pdf;         //< PDF value for this sample
    // float2     uv;
};

}

//
// TODO: these will become rtCallableBuffers soon (once I implement in optix)
//

rtDeclareVariable( unsigned, legionLightCount, , );

// LightSample legionLightSample( float2 seed, float3 shading_point, float3 shading_normal )
rtCallableProgram( legion::LightSample, legionLightSample_0, ( float2, float3, float3 ) ); 
rtCallableProgram( legion::LightSample, legionLightSample_1, ( float2, float3, float3 ) ); 
rtCallableProgram( legion::LightSample, legionLightSample_2, ( float2, float3, float3 ) ); 
rtCallableProgram( legion::LightSample, legionLightSample_3, ( float2, float3, float3 ) ); 
rtCallableProgram( legion::LightSample, legionLightSample_4, ( float2, float3, float3 ) ); 

// float3 legionLightEmission( float3 w_in, float light_dist, float3 light_normal  )
rtCallableProgram( float3, legionLightEvaluate_0, ( float3, float, float3 ) ); 
rtCallableProgram( float3, legionLightEvaluate_1, ( float3, float, float3 ) ); 
rtCallableProgram( float3, legionLightEvaluate_2, ( float3, float, float3 ) ); 
rtCallableProgram( float3, legionLightEvaluate_3, ( float3, float, float3 ) ); 
rtCallableProgram( float3, legionLightEvaluate_4, ( float3, float, float3 ) ); 

RT_CALLABLE_PROGRAM
legion::LightSample nullLightSample( float2, float3, float3 )
{ 
    legion::LightSample ls; 
    ls.w_in = ls.normal = make_float3( 0.0f );
    ls.distance = ls.pdf = 0.0f;
    return ls;
}

RT_CALLABLE_PROGRAM
float3 nullLightEvaluate( legion::LightSample )
{ 
    return make_float3( 0.0f );
} 


namespace legion
{

LDEVICE inline
LightSample lightSample( unsigned light_idx, float2 seed, float3 shading_point, float3 shading_normal )
{
    switch( light_idx )
    {
        case 0:   return legionLightSample_0( seed, shading_point, shading_normal );
        case 1:   return legionLightSample_1( seed, shading_point, shading_normal );
        case 2:   return legionLightSample_2( seed, shading_point, shading_normal );
        case 3:   return legionLightSample_3( seed, shading_point, shading_normal );
        default:  return legionLightSample_4( seed, shading_point, shading_normal );
    };
}


LDEVICE inline
float3 lightEvaluate( unsigned light_idx, float3 w_in, float light_dist, float3 light_normal )
{
    switch( light_idx )
    {
        case 0:  return legionLightEvaluate_0( w_in, light_dist, light_normal );
        case 1:  return legionLightEvaluate_1( w_in, light_dist, light_normal );
        case 2:  return legionLightEvaluate_2( w_in, light_dist, light_normal );
        case 3:  return legionLightEvaluate_3( w_in, light_dist, light_normal );
        default: return legionLightEvaluate_4( w_in, light_dist, light_normal );
    };
}

}

#endif // LEGION_OBJECTS_SURFACE_CUDA_LIGHT_HPP_

