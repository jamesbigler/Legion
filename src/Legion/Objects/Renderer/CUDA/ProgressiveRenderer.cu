
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
#include <Legion/Objects/Renderer/CUDA/Renderer.hpp>
#include <optixu/optixu_math_namespace.h>


rtDeclareVariable( unsigned, sample_index, , );
rtDeclareVariable( unsigned, samples_per_pass, , );

rtBuffer<float4, 2> output_buffer;



RT_PROGRAM void progressiveRendererRayGen()
{

    float3 result = make_float3( 0.0f );

    for( unsigned current_sample_index = sample_index;
         current_sample_index < sample_index+samples_per_pass;
         ++current_sample_index )
    {
        float2 screen_sample;
        float2 lens_sample;
        float  time_sample;
        const legion::uint64 sobol_index = 
            legion::generateSobolSamples( 
                launch_dim,
                launch_index,
                current_sample_index,
                screen_sample,
                lens_sample,
                time_sample );
        
        legion::RayGeometry rg = legionCameraCreateRay( lens_sample,
                                                        screen_sample,
                                                        time_sample );

        result += legion::radiance( sobol_index, rg.origin, rg.direction );
    }

    const float  spp    = static_cast<float>( samples_per_pass );
    const float4 prev   = output_buffer[ launch_index ];
    const float4 cur    = make_float4( result / spp, 1.0f );
    const float4 final  = optix::lerp( prev, cur, spp / ( spp+sample_index ) );
    output_buffer[ launch_index ] = final;
}
