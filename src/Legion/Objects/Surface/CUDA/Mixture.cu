
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
#include <Legion/Objects/Surface/CUDA/Surface.hpp>
#include <Legion/Objects/Texture/CUDA/Texture.hpp>
#include <Legion/Common/Math/CUDA/ONB.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>

//rtDeclareVariable( float3, reflectance, , );


legionDeclareTexture( float, mixture_weight);
legionDeclareSurface( surf0 );
legionDeclareSurface( surf1 );



RT_CALLABLE_PROGRAM
legion::BSDFSample mixtureSampleBSDF( 
        float3 seed,
        float3 w_out,
        legion::LocalGeometry p )
{
    const float pr = legionTex( mixture_weight, p, w_out );

    if( seed.z < pr )
    {
        const float pr_inv = 1.0f / pr;
        seed.z *= pr_inv;
        legion::BSDFSample sample = legionSampleBSDF( surf1, seed, w_out, p);
        const float4 fpdf0 = legionEvaluateBSDF( surf0, w_out, p, sample.w_in );
        const float pdf0            = fpdf0.w; 
        const float pdf1            = sample.pdf;
        const float pdf_combined    = optix::lerp( pdf0, pdf1, pr ); 

        const float3 f_pdf0         =  pdf0 <= 0.0f ? make_float3( 0.0f ) :
            make_float3( fpdf0 ) * 
            ( optix::dot( sample.w_in, p.shading_normal ) / pdf0 );
        const float3 f_pdf1         = sample.f_over_pdf;
        const float3 f_pdf_combined = optix::lerp( f_pdf0, f_pdf1, pr ); 

        sample.pdf        = pdf_combined * pr_inv; 
        sample.f_over_pdf = f_pdf_combined * pr_inv; 

        return sample;
    }
    else
    {
        const float pr_inv = 1.0f / ( 1.0f - pr );
        seed.z  = (seed.z - pr) * pr_inv;
        legion::BSDFSample sample  = legionSampleBSDF( surf0, seed, w_out, p );
        const float4 fpdf1 = legionEvaluateBSDF( surf0, w_out, p, sample.w_in );

        const float pdf0            = sample.pdf;
        const float pdf1            = fpdf1.w;
        const float pdf_combined    = optix::lerp( pdf0, pdf1, pr ); 
        
        const float3 f_pdf0         = sample.f_over_pdf;
        const float3 f_pdf1         =  pdf1 <= 0.0f ? make_float3( 0.0f ) :
            make_float3( fpdf1 ) * 
            ( optix::dot( sample.w_in, p.shading_normal ) / pdf1 );
        const float3 f_pdf_combined = optix::lerp( f_pdf0, f_pdf1, pr ); 

        sample.pdf        = pdf_combined * pr_inv; 
        sample.f_over_pdf = f_pdf_combined * pr_inv; 

        return sample;
    }
}


RT_CALLABLE_PROGRAM
float4 mixtureEvaluateBSDF(
        float3 w_out,
        legion::LocalGeometry p,
        float3 w_in )
{
    const float pr = legionTex( mixture_weight, p, w_out );
    //const float pr = 0.5f; 

    const float4 bsdf0 = legionEvaluateBSDF( surf0, w_out, p, w_in );
    const float4 bsdf1 = legionEvaluateBSDF( surf1, w_out, p, w_in );
    return optix::lerp( bsdf0, bsdf1, pr ); 
}


RT_CALLABLE_PROGRAM
float mixturePDF( float3 w_out, legion::LocalGeometry p, float3 w_in )
{
    const float pr = legionTex( mixture_weight, p, w_out );
    //const float pr = 0.5f; 
    const float pdf0 = legionPDF( surf0, w_out, p, w_in );
    const float pdf1 = legionPDF( surf1, w_out, p, w_in );
    return optix::lerp( pdf0, pdf1, pr ); 
}
