
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

/// \file IGeometry.hpp
/// Pure virtual interface for Geometry classes


#include <Legion/Objects/cuda_common.hpp>


rtDeclareVariable( float3, center, , );
rtDeclareVariable( float , radius, , );

// TODO: attrs should be in a header which can be included by all clients
rtDeclareVariable( legion::LocalGeometry, lgeom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );


RT_PROGRAM void sphereIntersect( int )
{
    // TODO: rename these origin, direction
    float3 O = ray.origin - center;
    float3 D = ray.direction;

    float b = optix::dot(O, D);
    float c = optix::dot(O, O)-radius*radius;
    float disc = b*b-c;
    if(disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);

        bool do_refine = false;

        float root11 = 0.0f;

        // refine root1
        float3 O1 = O + root1 * ray.direction;
        b = optix::dot(O1, D);
        c = optix::dot(O1, O1) - radius*radius;
        disc = b*b - c;

        if(disc > 0.0f) {
            sdisc = sqrtf(disc);
            root11 = (-b - sdisc);
        }

        bool check_second = true;
        if( rtPotentialIntersection( root1 + root11 ) ) {

            const float  t      = root1 + root11;
            const float3 normal = (O + t*D)/radius;

            // Fill in a localgeometry
            legion::LocalGeometry lg;
            lg.position_object  = ray.origin + t*ray.direction;
            lg.geometric_normal = normal;
            lg.shading_normal   = normal;
            lg.texcoord         = make_float2( 0.0f );

            lgeom = lg;

            if(rtReportIntersection(0))
                check_second = false;
        } 

        if(check_second) {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            if( rtPotentialIntersection( root2 ) ) {
                const float  t      = root2; 
                const float3 normal = (O + t*D)/radius;

                // Fill in a localgeometry
                legion::LocalGeometry lg;
                lg.position_object  = ray.origin + t*ray.direction;
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

