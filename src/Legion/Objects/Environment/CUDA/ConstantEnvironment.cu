
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

rtDeclareVariable( float3, radiance, , );

RT_CALLABLE_PROGRAM
float3 constantEnvironmentMissEvaluate( 
        float3 dir ) 
{
    return radiance; 
}


RT_CALLABLE_PROGRAM
float3 constantEnvironmentLightEvaluate( legion::LightSample )
{
    return radiance; 
}


RT_CALLABLE_PROGRAM 
legion::LightSample constantEnvironmentSample( 
        float2 sample_seed,
        float3 shading_point,
        float3 shading_normal )
{
    legion::LightSample sample;

    // TODO: sample hemisphere instead

    legion::uniformSampleSphere( sample_seed, sample.w_in, sample.pdf );
    sample.distance = 1e8f; // TODO: magic number
    sample.normal   = -sample.w_in;

    return sample; 
}
