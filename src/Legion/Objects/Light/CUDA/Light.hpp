
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



//float legionLightPDF( float3 w_in, float3 p )
rtCallableProgram( float, legionLightPDF, ( float3, float3 ) ); 


// LightSample legionLightSample( float2 seed, float3 shading_point, float3 shading_normal )
rtBuffer< rtCallableProgramId<legion::LightSample,float2,float3,float3>, 1 > legionLightSampleFuncs;

// float3 legionLightEmission( legion::LightSample light_info )
rtBuffer< rtCallableProgramId<float3,legion::LightSample>,1> legionLightEvaluateFuncs;

namespace legion
{

LDEVICE inline
unsigned lightCount()
{
    return legionLightSampleFuncs.size();
}


LDEVICE inline
LightSample lightSample( unsigned light_idx, float2 seed, float3 shading_point, float3 shading_normal )
{
    return legionLightSampleFuncs[light_idx]( seed, shading_point, shading_normal );
}


LDEVICE inline
float3 lightEvaluate( unsigned light_idx, legion::LightSample sample )
{
    return legionLightEvaluateFuncs[light_idx]( sample );
}


}

#endif // LEGION_OBJECTS_SURFACE_CUDA_LIGHT_HPP_

