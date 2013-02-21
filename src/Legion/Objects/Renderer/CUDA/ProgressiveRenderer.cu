
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
rtDeclareVariable( unsigned, do_byte_updates,   , );
rtDeclareVariable( unsigned, do_byte_complete,  , );

rtBuffer<float4, 2> float_output_buffer;
rtBuffer<uchar4, 2> byte_output_buffer;

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

    const float3 result =
        legion::radiance( sobol_index, rg.origin, rg.direction, light_index, sample_index );

    float3 pix_val = result;
    if( sample_index > 0 )
    {
        const float3 prev = make_float3( float_output_buffer[ launch_index ] );
        pix_val = optix::lerp( prev, result, 1.0f/sample_index );
    }

    float_output_buffer[ launch_index ] = make_float4( pix_val, 1.0f );

    //
    // Performa gamma correction and reinhard tonemapping for displayable buffer
    // TODO: allow user to control gamma, exposure and tonemapping algorithm
    //
    if( do_byte_updates  || 
      ( do_byte_complete && sample_index+1 == samples_per_pixel ) )
    {
        const float3 mapped =
            legion::gammaCorrect( 
                    legion::reinhardToneOperator( pix_val ), 2.2f
                    );
        byte_output_buffer[ launch_index ] = 
            make_uchar4( 255, mapped.x * 255, mapped.y*255, mapped.z*255 ); 
    }
}
