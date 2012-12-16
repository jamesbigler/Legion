
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


/// \file ILight.hpp
/// Pure virtual interface for all Light classes

#ifndef LEGION_INTERFACE_ILIGHTSHADER_HPP_
#define LEGION_INTERFACE_ILIGHTSHADER_HPP_




namespace legion
{

class VariableContainer;

/// Pure virtual interface for all LightShader classes
class ILight
{
public:

    
    virtual void setVariables( VariableContainer& container ) const = 0;


    /*
    /// Destroy an ILightShader object
    virtual         ~ILightShader();

    virtual         isEmitter()const=0;

    virtual bool    isSingular()const=0;

    // light_p is the geometry of the light, w_in is the incoming direction 
    // to that point on the light
    virtual Color   emittance( const LocalGeometry& light_geom,
                               const Vector3& w_in );

    void sample( const Vector2&       seed,
                 const LocalGeometry& p,
                 Vector3&             on_light,
                 float&               pdf );
    */
};


}

#endif // LEGION_INTERFACE_ILIGHTSHADER_HPP_
