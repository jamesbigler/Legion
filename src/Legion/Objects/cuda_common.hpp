
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

#ifndef LEGION_OBJECTS_CUDA_COMMON_HPP_
#define LEGION_OBJECTS_CUDA_COMMON_HPP_

//------------------------------------------------------------------------------
//
// this struct is to be used as the per-ray-data for radiance rays
//
//------------------------------------------------------------------------------
struct RadiancePRD 
{
  float3 result;
  float  importance;
  int    depth;
};

//------------------------------------------------------------------------------
//
// this struct is to be used as the per-ray-data for shadow rays
//
//------------------------------------------------------------------------------
struct ShadowPRD 
{
  float3 attenuation;
};


struct ShadowPRD 

//------------------------------------------------------------------------------
//
// Common rtVariables.  These will all be set internally by legion 
//
//------------------------------------------------------------------------------

// TODO: document these
rtDeclareVariable( unsigned, legion_radiance_ray_type, , );
rtDeclareVariable( unsigned, legion_shadow_ray_type  , , );
rtDeclareVariable( rtObject, legion_top_group, , );

rtCallableProgram( legion::RayGeometry,
                   legionCameraCreateRay, 
                   (float2, float2, float ) );


#endif // LEGION_OBJECTS_CUDA_COMMON_HPP_
