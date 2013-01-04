
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


/// \file PointLight.hpp
/// Pure virtual interface for all Light classes

#ifndef LEGION_INTERFACE_POINT_LIGHTSHADER_HPP_
#define LEGION_INTERFACE_POINT_LIGHTSHADER_HPP_


#include <Legion/Objects/Light/ILight.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Vector.hpp>

namespace legion
{


/// Pure virtual interface for all LightShader classes
class PointLight : public ILight
{
public:

    void setVariables( VariableContainer& container ) const;
    
    void setIntensity( const Color& intensity );
    void setPosition( const Vector3& position );

private:
    Color   m_intensity;
    Vector3 m_position;
};


}

#endif // LEGION_INTERFACE_POINT_LIGHTSHADER_HPP_
