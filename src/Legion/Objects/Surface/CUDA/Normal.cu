
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


#include <Legion/Objects/Surface/CUDA/Surface.hpp>
#include <optixu/optixu_math_namespace.h>
#include <optix.h>


rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 


rtDeclareVariable( RadiancePRD, radiance_prd, rtPayload, );
rtDeclareVariable( ShadowPRD,   shadow_prd,   rtPayload, );


RT_PROGRAM void normalAnyHit()
{
  // this material is opaque, so it fully attenuates all shadow rays
  shadow_prd.attenuation = make_float3(0);

  rtTerminateRay();
}

RT_PROGRAM void normalClosestHit()
{
    radiance_prd.result = optix::normalize( 
            rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal) 
            ) * 0.5f + 0.5f;
}
