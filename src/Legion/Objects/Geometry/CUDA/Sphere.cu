
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


// TODO: clean this up, share intersection code, etc

#include <Legion/Objects/cuda_common.hpp>
#include <Legion/Common/Math/CUDA/ONB.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>


rtDeclareVariable( float3, center, , );
rtDeclareVariable( float , radius, , );

// TODO: attrs should be in a header which can be included by all clients
rtDeclareVariable( legion::LocalGeometry, lgeom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );

__device__ __inline__
bool sphereIntersectImpl( 
        float3 origin,
        float3 direction, 
        float3 center, 
        float  radius,
        float  tmin,
        float  tmax,
        legion::LocalGeometry& sample )
{
    const float3 temp = origin - center;
    const float  twoa = 2.0f*optix::dot( direction, direction );
    const float  b    = 2.0f*optix::dot( direction, temp );
    const float  c    = optix::dot( temp, temp ) - radius*radius;

    float  discriminant = b*b- 2.0f*twoa*c;

    if( discriminant > 0.0f )
    {
        discriminant = sqrtf( discriminant );
        float t = (-b - discriminant) / twoa;
        if (t < tmin) t = (-b + discriminant) / twoa;

        if (t >= tmin && t <= tmax) 
        {
            // we have a hit -- populate hit record
            sample.position         = origin + t*direction;
            sample.geometric_normal = (sample.position - center) / radius;
            sample.shading_normal   = sample.geometric_normal;
            sample.texcoord         = make_float2( 0.0f );
            return true;
        }
    }
    return false;
}

__device__ __inline__
bool sphereIntersectImpl2( 
        float3 origin,
        float3 direction, 
        float3 center, 
        float  radius,
        float  tmin,
        float  tmax,
        legion::LocalGeometry& lg )
{
    float3 O = origin - center;
    float3 D = direction;

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

        float  t = root1 + root11;
        if( t >= tmin && t <= tmax  ) {

            const float3 normal = (O + t*D)/radius;

            // Fill in a localgeometry
            lg.position         = origin + t*direction;
            lg.geometric_normal = normal;
            lg.shading_normal   = normal;
            lg.texcoord         = make_float2( 0.0f );

            return true; 
        } 

        float root2 = (-b + sdisc) +  root1;
        t = root2;
        if( t >= tmin && t <= tmax  ) {

            const float3 normal = (O + t*D)/radius;

            // Fill in a localgeometry
            lg.position         = origin + t*direction;
            lg.geometric_normal = normal;
            lg.shading_normal   = normal;
            lg.texcoord         = make_float2( 0.0f );

            return true; 
        }
    }
    return false;
}


RT_PROGRAM void sphereIntersect( int )
{
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
        if( rtPotentialIntersection( root1 + root11 ) ) {

            const float  t      = root1 + root11;
            const float3 normal = (O + t*D)/radius;

            // Fill in a localgeometry
            legion::LocalGeometry lg;
            lg.position         = ray.origin + t*ray.direction;
            lg.geometric_normal = normal;
            lg.shading_normal   = normal;
            lg.texcoord         = make_float2( 0.0f );

            lgeom = lg;

            if(rtReportIntersection(0))
                check_second = false;
        } 

        if(check_second) {

            float root2 = (-b + sdisc) +  root1;
            if( rtPotentialIntersection( root2 ) ) {

                const float  t      = root2; 
                const float3 normal = (O + t*D)/radius;

                // Fill in a localgeometry
                legion::LocalGeometry lg;
                lg.position         = ray.origin + t*ray.direction;
                lg.geometric_normal = normal;
                lg.shading_normal   = normal;
                lg.texcoord         = make_float2( 0.0f );

                lgeom = lg;

                rtReportIntersection(0);
            }
        }
    }
}


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
__device__ __inline__
bool sphereIntersectI( 
        float3 origin,
        float3 direction, 
        float3 center, 
        float  radius,
        float  tmin,
        float  tmax,
        legion::LocalGeometry& sample )
{
    const float3 temp = origin - center;
    const float  twoa = 2.0f*optix::dot( direction, direction );
    const float  b    = 2.0f*optix::dot( direction, temp );
    const float  c    = optix::dot( temp, temp ) - radius*radius;

    float  discriminant = b*b- 2.0f*twoa*c;

    if( discriminant > 0.0f )
    {
        discriminant = sqrtf( discriminant );
        float t = (-b - discriminant) / twoa;
        if (t < tmin) t = (-b + discriminant) / twoa;

        if (t >= tmin && t <= tmax) 
        {
            // we have a hit -- populate hit record
            sample.position         = origin + t*direction;
            sample.geometric_normal = (sample.position - center) / radius;
            sample.shading_normal   = sample.geometric_normal;
            sample.texcoord         = make_float2( 0.0f );
            return true;
        }
    }
    return false;
}

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
    float3 w_in = make_float3( cos_phi*sin_theta, sin_phi*sin_theta, cos_theta);
    w_in = uvw.inverseTransform( w_in );

    // TODO: magic numbers
    if( !sphereIntersectImpl(
                shading_point,
                w_in, center,
                radius,
                0.0f,
                1e18f,
                sample.point_on_light ) )
        sample.pdf = 0.0f;

    // TODO: optimize
    w_in = ( shading_point - sample.point_on_light.position );
    if( w_in.x == 0 && w_in.y == 0 && w_in.z == 0 )
        sample.pdf = 0.0f;

    //if( optix::length( shading_point - sample.point_on_light.position ) == 0.0f )
    //    sample.pdf = 0.0f;
    return sample;
}

