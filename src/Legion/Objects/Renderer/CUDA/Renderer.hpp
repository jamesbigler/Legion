
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

namespace legion
{

__device__
float3 radiance( uint64 sobol_index, float3 origin, float3 direction, unsigned light_idx )
{
    legion::RadiancePRD prd;
    prd.radiance            = make_float3( 0.0f );
    prd.attenuation         = make_float3( 1.0f );
    prd.pdf                 = 1.0f;
    prd.sobol_index         = sobol_index; 
    prd.sobol_dim           = 5u; 
    prd.use_mis_weight      = false;
    prd.done                = true;
    prd.light_index         = light_idx;

    float3 radiance    = make_float3( 0.0f );
    float3 attenuation = make_float3( 1.0f );
    optix::Ray ray = legion::makePrimaryRay( origin, direction );
    
    for( unsigned i = 0u; i < 4; ++i ) 
    {
        prd.done     = true;
        prd.radiance = make_float3( 0.0f );
        rtTrace( legion_top_group, ray, prd );
        
        radiance += prd.radiance * attenuation;

        if( prd.done || attenuation.x + attenuation.y + attenuation.z < 0.001f )
            break;
        attenuation *= prd.attenuation;
        /*
        const float    p_continue          = fmaxf( attenuation );
        const unsigned RR_SOBOL_DIM_OFFSET = 64;
        if( legion::Sobol::gen( sobol_index, RR_SOBOL_DIM_OFFSET+i ) > p_continue  )
            break;

        attenuation   /= p_continue;
        */
        ray.direction  = prd.direction;
        ray.origin     = prd.origin;
    }

    return radiance;
}

}
