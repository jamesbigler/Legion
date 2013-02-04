
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

/// \file ISurface.hpp
/// Pure virtual interface for Surface Shaders
#ifndef LEGION_OBJECTS_SURFACE_ISURFACE_HPP_
#define LEGION_OBJECTS_SURFACE_ISURFACE_HPP_

#include <Legion/Objects/IObject.hpp>


namespace legion
{

/// Pure virtual interface for Surfaces
class ISurface : public IObject
{
public:
    ISurface( Context* context ) : IObject( context ) {}

    virtual ~ISurface() {}


    virtual const char* name()const=0;
    virtual const char* sampleBSDFFunctionName()const=0;
    virtual const char* evaluateBSDFFunctionName()const=0;
    // TODO: make this bsdfPDFFunctionName
    virtual const char* pdfFunctionName()const=0;
    virtual const char* emissionFunctionName()const=0;

    
    /// Return the name of this Surface type
    /// static std::string name();

    /// Return the name of this Camera's cuda ray generation function 
    /// static std::string rayGenFunctionName();


    /*
    /// Destroy ISurface object
    virtual         ~ISurface();

    /// Sample the BSDF at given local geometry.
    ///   \param      seed   2D sampling seed in [0,1]^2
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param[out] w_in   Direction to light
    ///   \param[out] pdf    The BSDF PDF for w_in 
    virtual void    sampleBSDF( const Vector2& seed,
                                const Vector3& w_out,
                                const LocalGeometry& p,
                                Vector3& w_in,
                                Color& f_over_pdf )const=0;

    virtual bool    isSingular()const=0;


    /// Compute the PDF value for the given incoming/outgoing direction pair
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param      w_in   Direction to light
    ///   \returns The pdf
    virtual float   pdf( const Vector3& w_out,
                         const LocalGeometry& p,
                         const Vector3& w_in )const=0;


    /// Compute the BSDF value for the given incoming/outgoing direction pair
    ///   \param      w_out  Outgoing light direction (-ray.direction typically)
    ///   \param      p      Local geometry information of point being shaded
    ///   \param      w_in   Direction to light
    ///   \returns The value of the bsdf
    virtual Color   evaluateBSDF( const Vector3& w_out,
                                  const LocalGeometry& p,
                                  const Vector3& w_in )const=0;
    */
};


}

#endif // LEGION_OBJECTS_SURFACE_ISURFACE_HPP_
