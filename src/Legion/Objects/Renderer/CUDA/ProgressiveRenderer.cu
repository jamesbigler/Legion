
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


//#include <Legion/Objects/Camera/CUDA/Camera.hpp>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optix.h>

rtDeclareVariable( optix::Matrix4x4, camera_to_world, , );
rtDeclareVariable( float           , focal_distance , , );
rtDeclareVariable( float           , aperture_radius, , );
rtDeclareVariable( float4          , view_plane     , , );


__device__ __forceinline__ 
  legion::RayGeometry thinLensCreateRay(
        float2 aperture_sample,
        float2 screen_sample,
        float  time )
{
    const float2 disk_sample = optix::square_to_disk( aperture_sample );
    legion::RayGeometry r;
    r.origin      = make_float3( aperture_radius * disk_sample, 0.0f );
    const float3 on_plane = make_float3(
            optix::lerp( view_plane.x, view_plane.y, screen_sample.x ),
            optix::lerp( view_plane.w, view_plane.z, screen_sample.y ),
            -focal_distance
            );
    r.direction = on_plane - r.origin;

    r.origin = make_float3( camera_to_world*make_float4( r.origin, 1.0f ) );
    r.direction = make_float3( camera_to_world*make_float4(r.direction, 0.0f) );
    r.direction = optix::normalize( r.direction );
    return r;
}



//#define TIME_VIEW
#ifdef TIME_VIEW
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;
#endif

using namespace optix;

//#define PIXEL_X 1124
//#define PIXEL_Y 330
#define PIXEL_X 858
#define PIXEL_Y 237

RT_PROGRAM void progressiveRendererRayGen()
{
#ifdef TIME_VIEW
  clock_t t0 = clock(); 
#endif
  //if (launch_index != make_uint2(715,594))
  //if (launch_index != make_uint2(PIXEL_X,PIXEL_Y))
  //  return;
  //if ((launch_index.x/1) != (PIXEL_X/1) || (launch_index.y/1) != (PIXEL_Y/1))
  //  return;

    const unsigned light_index = light_seed * legion::lightCount();
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
        
    legion::RayGeometry rg = thinLensCreateRay( lens_sample,
    //legion::RayGeometry rg = legionCameraCreateRay( lens_sample,
            screen_sample,
            time_sample );

    // printf("64 bit = 0x%llx\n", 1337ull);
    // printf("32 bit = 0x%x\n", 1338);
    unsigned int num_iters;
    const float3 result =
        legion::radiance(
                sobol_index,
                rg.origin,
                rg.direction,
                light_index,
                sample_index,
                num_iters);

    float3 pix_val = result;
    if( sample_index > 0 )
    {
        const float3 prev = make_float3( float_output_buffer[ launch_index ] );
        pix_val = optix::lerp( prev, result, 1.0f/sample_index );
    }

#ifdef TIME_VIEW
  clock_t t1 = clock(); 
 
  float expected_fps   = 1.0f;
  float pixel_time     = ( t1 - t0 ) * time_view_scale * expected_fps;
#if 0
    float_output_buffer[ launch_index ] = make_float4( make_float3(  pixel_time ), 1.0f );
#else
    float_output_buffer[ launch_index ] = make_float4( 100,0,100, 1.0f );
    if (launch_index == make_uint2(PIXEL_X,PIXEL_Y)) printf("\n(%u,%u): delta_t = %ld\n", launch_index.x, launch_index.y, t1-t0);
    if (launch_index == make_uint2(PIXEL_X,PIXEL_Y)) printf("\n(%u,%u): num_iters = %u\n", launch_index.x, launch_index.y, num_iters);

#endif
#else
    float_output_buffer[ launch_index ] = make_float4( pix_val, 1.0f );
#endif
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
