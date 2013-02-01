
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
#include <Legion/Objects/Light/CUDA/Light.hpp>
#include <Legion/Objects/Renderer/CUDA/Renderer.hpp>
#include <optixu/optixu_math_namespace.h>


rtDeclareVariable( float,    light_seed,        , );
rtDeclareVariable( unsigned, sample_index,      , );
rtDeclareVariable( unsigned, samples_per_pixel, , );

rtBuffer<float4, 2> output_buffer;

RT_PROGRAM void progressiveRendererRayGen()
{
    const unsigned light_index = light_seed*legionLightCount;
    float2 screen_sample;
    float2 lens_sample;
    float  time_sample;
    const  legion::uint64 sobol_index = 
        legion::generateSobolSamples( 
                launch_dim,
                launch_index,
                sample_index,
                screen_sample,
                lens_sample,
                time_sample );
        
    legion::RayGeometry rg = legionCameraCreateRay( lens_sample,
            screen_sample,
            time_sample );

    const float3 result = legion::radiance( sobol_index, rg.origin, rg.direction, light_index );

    const float4 cur = make_float4( result, 1.0f );
    if( sample_index == 0 )
    {
        output_buffer[ launch_index ] = cur;
    }
    else
    {
        const float4 prev   = output_buffer[ launch_index ];
        const float4 final  = optix::lerp( prev, cur, 1.0f / sample_index );
        output_buffer[ launch_index ] = final;
    }
}
