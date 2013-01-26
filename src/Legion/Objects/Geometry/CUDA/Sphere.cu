
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
#include <Legion/Common/Math/CUDA/ONB.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>

rtDeclareVariable( float3, center, , );
rtDeclareVariable( float , radius, , );

rtDeclareVariable( legion::LocalGeometry, local_geom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );


//-----------------------------------------------------------------------------
//
// Helpers
//
//-----------------------------------------------------------------------------

struct IntersectReporter
{
    __device__ __inline__ bool check_t( float t ) { return rtPotentialIntersection( t ); }
    __device__ __inline__ bool report ( float t, legion::LocalGeometry lg ) 
    { 
        local_geom = lg;
        return rtReportIntersection( t ); 
    }
};


struct SampleReporter
{
    __device__ __inline__ SampleReporter( legion::LocalGeometry& lg ) : local_geom( lg ) {}
    __device__ __inline__ bool check_t( float t )
    { return t > 0.0001f; }

    __device__ __inline__ bool report ( float t, legion::LocalGeometry lg ) 
    { local_geom = lg; return true; }

    legion::LocalGeometry& local_geom;
};


template <typename Reporter>
static __device__ __inline__
bool sphereIntersectImpl( 
        float3 origin,
        float3 direction, 
        float3 center, 
        float  radius,
        Reporter& reporter )
{
    float3 O = origin - center;
    float3 D = direction;

    float b = optix::dot(O, D);
    float c = optix::dot(O, O)-radius*radius;
    float disc = b*b-c;
    
    bool intersection_found = false;

    if(disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);


        float root11 = 0.0f;

        // refine root1
        if( fabsf(root1) > 10.f * radius )
        {
            float3 O1 = O + root1 * direction;
            b = optix::dot(O1, D);
            c = optix::dot(O1, O1) - radius*radius;
            disc = b*b - c;

            if(disc > 0.0f)
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        const float t = root1 + root11;
        if( reporter.check_t( t ) ) 
        {

            const float3 normal = (O + t*D)/radius;

            // Fill in a localgeometry
            legion::LocalGeometry lg;
            lg.position         = origin + t*direction;
            lg.geometric_normal = normal;
            lg.shading_normal   = normal;
            lg.texcoord         = make_float2( 0.0f );

            intersection_found = reporter.report( t, lg );
        } 

        if( !intersection_found )
        {
            const float t = (-b + sdisc) +  root1;
            if( reporter.check_t( t ) ) 
            {
                const float3 normal = (O + t*D)/radius;

                // Fill in a localgeometry
                legion::LocalGeometry lg;
                lg.position         = origin + t*direction;
                lg.geometric_normal = normal;
                lg.shading_normal   = normal;
                lg.texcoord         = make_float2( 0.0f );

                intersection_found = reporter.report( t, lg );
            }
        }
    }

    return intersection_found;
}


//-----------------------------------------------------------------------------
//
//
//
//-----------------------------------------------------------------------------
RT_PROGRAM void sphereIntersect( int )
{
    /*
    // TODO: Bug in optix intersection inlining is breaking this
    IntersectReporter reporter;
    sphereIntersectImpl<IntersectReporter>(
                ray.origin,
                ray.direction,
                center,
                radius,
                reporter );
    */

    float3 O = ray.origin - center;
    float3 D = ray.direction;

    float b = optix::dot(O, D);
    float c = optix::dot(O, O)-radius*radius;
    float disc = b*b-c;
    if(disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);


        float root11 = 0.0f;

        // refine root1
        if( fabsf(root1) > 10.f * radius )
        {
            float3 O1 = O + root1 * ray.direction;
            b = optix::dot(O1, D);
            c = optix::dot(O1, O1) - radius*radius;
            disc = b*b - c;

            if(disc > 0.0f)
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        bool check_second = true;
        const float t = root1 + root11;
        if( rtPotentialIntersection( t ) ) {

            const float3 normal = (O + t*D)/radius;

            // Fill in a localgeometry
            legion::LocalGeometry lg;
            lg.position         = ray.origin + t*ray.direction;
            lg.geometric_normal = normal;
            lg.shading_normal   = normal;
            lg.texcoord         = make_float2( 0.0f );

            local_geom = lg;

            if(rtReportIntersection(0))
                check_second = false;
        } 

        if(check_second) {

            const float t = (-b + sdisc) +  root1;
            if( rtPotentialIntersection( t ) ) {

                const float3 normal = (O + t*D)/radius;

                // Fill in a localgeometry
                legion::LocalGeometry lg;
                lg.position         = ray.origin + t*ray.direction;
                lg.geometric_normal = normal;
                lg.shading_normal   = normal;
                lg.texcoord         = make_float2( 0.0f );

                local_geom = lg;

                rtReportIntersection(0);
            }
        }
    }
}

//-----------------------------------------------------------------------------
//
//
//
//-----------------------------------------------------------------------------

RT_PROGRAM void sphereBoundingBox( int, float result[6] )
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if( radius > 0.0f  && !isinf(radius) ) {
    aabb->m_min = center - make_float3( radius );
    aabb->m_max = center + make_float3( radius );
  } else {
    aabb->invalidate();
  }
}

//-----------------------------------------------------------------------------
//
//
//
//-----------------------------------------------------------------------------

RT_CALLABLE_PROGRAM
legion::LightSample sphereSample( float2 sample_seed, float3 shading_point )
{
    legion::LightSample sample;
    sample.pdf = 0.0f;

    float3 temp = center - shading_point;
    float d = optix::length( temp );
    temp /= d;
    
    if ( d <= radius )
        return sample;

    // internal angle of cone surrounding light seen from viewpoint
    float sin_alpha_max = (radius / d);
    float cos_alpha_max = sqrtf( 1.0f - sin_alpha_max*sin_alpha_max );

    float q    = 2.0f*legion::PI*( 1.0f - cos_alpha_max ); // solid angle
    sample.pdf =  1.0f/q;                          // pdf is one / solid angle

    const float phi       = 2.0f*legion::PI*sample_seed.x;
    const float cos_theta = 1.0f - sample_seed.y * ( 1.0f - cos_alpha_max );
    const float sin_theta = sqrtf( 1.0f - cos_theta*cos_theta );
    const float cos_phi = cosf( phi );
    const float sin_phi = sinf( phi );

    legion::ONB uvw( temp );
    float3 w_in = optix::normalize( make_float3( cos_phi*sin_theta, sin_phi*sin_theta, cos_theta) );
    w_in = uvw.inverseTransform( w_in );

    SampleReporter reporter( sample.point_on_light );
    if( !sphereIntersectImpl<SampleReporter>(
                shading_point,
                w_in,
                center,
                radius,
                reporter ) )
        sample.pdf = 0.0f;

    return sample;
}

