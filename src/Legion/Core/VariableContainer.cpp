
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

#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Core/Color.hpp>

using namespace legion;

namespace
{
    optix::Variable getVariable( optix::ScopedObj* scoped,
                                 const std::string& name )
    {
        optix::Variable v = scoped->queryVariable( name );
        if( !v )
            v = scoped->declareVariable( name );
        return v;
    }
}


VariableContainer::VariableContainer( optix::ScopedObj* scoped )
    : m_scoped( scoped )
{
}


void VariableContainer::setFloat( const std::string& name, float val ) const
{
    getVariable( m_scoped, name )->setFloat( val );
}


void VariableContainer::setFloat( const std::string& name, const Vector2& val ) const
{
    getVariable( m_scoped, name )->setFloat( val.x(), val.y() );
}


void VariableContainer::setFloat( const std::string& name, const Vector3& val )const
{
    getVariable( m_scoped, name )->setFloat( val.x(), val.y(), val.z() );
}


void VariableContainer::setFloat( const std::string& name, const Vector4& val ) const
{
    getVariable( m_scoped, name )->
        setFloat( val.x(), val.y(), val.z(), val.w() );
}


void VariableContainer::setFloat( const std::string& name, const Color& val ) const
{
    getVariable( m_scoped, name )->setFloat( val.r(), val.g(), val.b() );
}


void VariableContainer::setUnsigned( const std::string& name, unsigned val ) const
{
    getVariable( m_scoped, name )->setUint( val );
}


void VariableContainer::setMatrix( const std::string& name, const Matrix& val )  const
{
    getVariable( m_scoped, name )->setMatrix4x4fv( false, val.getData() );
}
    

void VariableContainer::setBuffer( const std::string& name, optix::Buffer val ) const
{
    getVariable( m_scoped, name )->set( val );
}
