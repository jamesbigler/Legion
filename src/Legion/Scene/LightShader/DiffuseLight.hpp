 
// Copyright (C) 2011 R. Keith Morley
//
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
// (MIT/X11 License)

/// \file DiffuseLight.hpp
/// DiffuseLight

#ifndef LEGION_DIFFUSELIGHT_HPP_
#define LEGION_DIFFUSELIGHT_HPP_

#include <Legion/Scene/LightShader/MeshLight.hpp>
#include <Legion/Core/Color.hpp>

namespace legion
{


class DiffuseLight : public MeshLight
{
public:
    DiffuseLight( Context* context, const std::string& name );
    
    ~DiffuseLight();

    Color power()const;

    Color emittance( const LocalGeometry& light_geom,
                     const Vector3& w_in )const;

    void setEmittance( const Color& color );

private:

    Color m_emittance;
};


}


#endif // LEGION_DIFFUSELIGHT_HPP_
