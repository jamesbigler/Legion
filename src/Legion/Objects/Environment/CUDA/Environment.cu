
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

#include <Legion/Objects/Environment/CUDA/Environment.hpp>
#include <Legion/Objects/Light/CUDA/Light.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>

rtDeclareVariable( optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void legionEnvironment()
{
    float w = 1.0f;
    if( radiance_prd.use_mis_weight )
    {
        const float light_pick_p = 1.0f / legionLightCount;
        const float pdf          = legion::ONE_DIV_PI * 0.25f;
        w = legion::powerHeuristic( radiance_prd.pdf, pdf*light_pick_p );
    }
    const float3 radiance = legionEnvironmentMissEvaluate( ray.direction );
    radiance_prd.radiance = w * radiance;
    radiance_prd.done     = true;
}
