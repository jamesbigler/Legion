
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

// TODO: attrs should be in a header which can be included by all clients
rtDeclareVariable( legion::LocalGeometry, lgeom, attribute local_geom, ); 
rtDeclareVariable( optix::Ray,            ray,   rtCurrentRay, );

rtDeclareVariable( float4, plane,  , );
rtDeclareVariable( float3, v1,     , );
rtDeclareVariable( float3, v2,     , );
rtDeclareVariable( float3, anchor, , );

RT_PROGRAM void parallelogramIntersect( int )
{
    const float3 n = make_float3( plane );
    const float dt = optix::dot(ray.direction, n );
    const float t  = ( plane.w - optix::dot(n, ray.origin ) ) / dt;

    if( t > ray.tmin && t < ray.tmax ) {
        const float3 p  = ray.origin + ray.direction * t;
        const float3 vi = p - anchor;
        float a1 = optix::dot(v1, vi);
        if(a1 >= 0 && a1 <= 1){
            float a2 = optix::dot(v2, vi);
            if(a2 >= 0 && a2 <= 1){
                if( rtPotentialIntersection( t ) ) {

                    // Fill in a localgeometry
                    legion::LocalGeometry lg;
                    //lg.position_object  = p;
                    lg.position         = p;
                    lg.geometric_normal = n;
                    lg.shading_normal   = n;
                    lg.texcoord         = make_float2( a1, a2 );
                    lgeom = lg;

                    rtReportIntersection( 0 );
                }
            }
        }
    }
}

RT_PROGRAM void parallelogramBoundingBox( int, float result[6] )
{
  // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
  const float3 tv1  = v1 / optix::dot( v1, v1 );
  const float3 tv2  = v2 / optix::dot( v2, v2 );
  const float3 p00  = anchor;
  const float3 p01  = anchor + tv1;
  const float3 p10  = anchor + tv2;
  const float3 p11  = anchor + tv1 + tv2;
  const float  area = optix::length( optix::cross(tv1, tv2) );
  
  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ) );
    aabb->m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ) );
  } else {
    aabb->invalidate();
  }
}


RT_CALLABLE_PROGRAM
float parallelogramPDF( float3 w_in, float3 p )
{

    /*
    // Intersect pgram here...
    // ...

    Vector3 to_light = localrec.p - rec.p;
    double dist      = to_light.makeUnitVector();
    double cosine    = -dot(localrec.uvw.w(),to_light);
    if ( cosine <= 0.0 )
        return 0.0;
    return inverse_area*dist*dist / cosine;
    */
    return 1.0f;
}
