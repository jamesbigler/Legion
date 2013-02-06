
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

#ifndef LEGION_COMMON_UTIL_PARAMETERS_HPP_
#define LEGION_COMMON_UTIL_PARAMETERS_HPP_

#include <Legion/Core/Color.hpp>
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <string>
#include <map>
#include <iosfwd>

namespace legion
{


class ITexture;


class Parameters
{
public:
    enum ParameterType
    {
        TYPE_FLOAT,
        TYPE_INT,
        TYPE_VECTOR,
        TYPE_COLOR,
        TYPE_MATRIX,
        TYPE_TEXTURE,
        TYPE_STRING
    };

    template <typename T>
    bool get( const std::string& name, const T& default_val, T& val )const;
   
    template <typename T>
    T get( const std::string& name, const T& default_val )const;
    
    bool get( const std::string& name, float&       val )const;
    bool get( const std::string& name, int&         val )const;
    bool get( const std::string& name, ITexture*&   val )const;
    bool get( const std::string& name, Vector3&     val )const;
    bool get( const std::string& name, Color&       val )const;
    bool get( const std::string& name, Matrix&      val )const;
    bool get( const std::string& name, std::string& val )const;

    bool set( const std::string& name, float              val );
    bool set( const std::string& name, int                val );
    bool set( const std::string& name, ITexture*          val );
    bool set( const std::string& name, const Vector3&     val );
    bool set( const std::string& name, const Color&       val );
    bool set( const std::string& name, const Matrix&      val );
    bool set( const std::string& name, const std::string& val );

    void reportUnused( std::ostream& out )const;

    void clear();

private:
    template <typename T>
    struct Param
    {
        Param( T val_ ) : val( val_ ), was_queried( false ) {}
        T    val;
        mutable bool was_queried;
    };

    typedef std::map<std::string, Param<float> >       FloatParams;
    typedef std::map<std::string, Param<int> >         IntParams;
    typedef std::map<std::string, Param<Vector3> >     VectorParams;
    typedef std::map<std::string, Param<Color> >       ColorParams;
    typedef std::map<std::string, Param<Matrix> >      MatrixParams;
    typedef std::map<std::string, Param<std::string> > StringParams;
    typedef std::map<std::string, Param<ITexture*> >   TextureParams;

    FloatParams     m_float_params;
    IntParams       m_int_params;
    VectorParams    m_vector_params;
    ColorParams     m_color_params;
    MatrixParams    m_matrix_params;
    StringParams    m_string_params;
    TextureParams   m_texture_params;
};


template <typename T>
T Parameters::get( const std::string& name, const T& default_val )const
{
    T val;
    get( name, default_val, val );
    return val;
}


template <typename T>
bool Parameters::get( 
        const std::string& name,
        const T&           default_val,
        T&                 val )const
{
    val = default_val;
    return get( name, val );
}

} // end namespace legion

#endif // LEGION_COMMON_UTIL_PARAMETERS_HPP_
