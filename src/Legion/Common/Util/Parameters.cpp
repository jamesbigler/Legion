
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


#include <Legion/Common/Util/Parameters.hpp>
#include <algorithm>

using namespace legion;


namespace
{
    // Templated for a ::value_type from any of the param maps
    template <typename T>
    struct ReportUnused
    {
        ReportUnused( std::ostream& o ) : out( o ) {}
        void operator()( const T& t )
        {
            if( !t.second.was_queried )
                out << "Unused parameter: " << t.first << std::endl; 
        }

        std::ostream& out;
    };
}



#define PARAMETERS_GET_IMPL( param_type, map_name, val_type )                  \
    bool Parameters::get( const std::string& name, val_type& val )const        \
{                                                                              \
    param_type::const_iterator it = map_name.find( name );                     \
    if( it == map_name.end() )                                                 \
        return false;                                                          \
                                                                               \
    it->second.was_queried = true;                                             \
    val = it->second.val;                                                      \
    return true;                                                               \
}

PARAMETERS_GET_IMPL( FloatParams,   m_float_params,   float       );
PARAMETERS_GET_IMPL( IntParams,     m_int_params,     int         );
PARAMETERS_GET_IMPL( Vector2Params, m_vector2_params, Vector2     );
PARAMETERS_GET_IMPL( Vector3Params, m_vector3_params, Vector3     );
PARAMETERS_GET_IMPL( Vector4Params, m_vector4_params, Vector4     );
PARAMETERS_GET_IMPL( ColorParams,   m_color_params,   Color       );
PARAMETERS_GET_IMPL( MatrixParams,  m_matrix_params,  Matrix      );
PARAMETERS_GET_IMPL( StringParams,  m_string_params,  std::string );
PARAMETERS_GET_IMPL( TextureParams, m_texture_params, ITexture*   );


#define PARAMETERS_SET_IMPL( map_name, val_type )                              \
    bool Parameters::set( const std::string& name, val_type val )              \
    {                                                                          \
        bool already_present = map_name.count( name );                         \
        map_name.insert( std::make_pair( name, Param<val_type>( val ) ) );     \
        return already_present;                                                \
    }

#define PARAMETERS_SET_REF_IMPL( map_name, val_type )                          \
    bool Parameters::set( const std::string& name, const val_type& val )       \
    {                                                                          \
        bool already_present = map_name.count( name );                         \
        map_name.insert( std::make_pair( name, Param<val_type>( val ) ) );     \
        return already_present;                                                \
    }

PARAMETERS_SET_IMPL    ( m_float_params,   float       );
PARAMETERS_SET_IMPL    ( m_int_params,     int         );
PARAMETERS_SET_IMPL    ( m_texture_params, ITexture*   );
PARAMETERS_SET_REF_IMPL( m_vector2_params, Vector2     );
PARAMETERS_SET_REF_IMPL( m_vector3_params, Vector3     );
PARAMETERS_SET_REF_IMPL( m_vector4_params, Vector4     );
PARAMETERS_SET_REF_IMPL( m_color_params,   Color       );
PARAMETERS_SET_REF_IMPL( m_matrix_params,  Matrix      );
PARAMETERS_SET_REF_IMPL( m_string_params,  std::string );


void Parameters::reportUnused( std::ostream& out )const
{
    std::for_each( m_float_params.begin(), m_float_params.end(), 
                   ReportUnused<FloatParams::value_type>( out ) );
    
    std::for_each( m_int_params.begin(), m_int_params.end(), 
                   ReportUnused<IntParams::value_type>( out ) );
    
    std::for_each( m_vector2_params.begin(), m_vector2_params.end(), 
                   ReportUnused<Vector2Params::value_type>( out ) );

    std::for_each( m_vector3_params.begin(), m_vector3_params.end(), 
                   ReportUnused<Vector3Params::value_type>( out ) );

    std::for_each( m_vector4_params.begin(), m_vector4_params.end(), 
                   ReportUnused<Vector4Params::value_type>( out ) );

    std::for_each( m_color_params.begin(), m_color_params.end(), 
                   ReportUnused<ColorParams::value_type>( out ) );

    std::for_each( m_matrix_params.begin(), m_matrix_params.end(), 
                   ReportUnused<MatrixParams::value_type>( out ) );

    std::for_each( m_string_params.begin(), m_string_params.end(), 
                   ReportUnused<StringParams::value_type>( out ) );

    std::for_each( m_texture_params.begin(), m_texture_params.end(), 
                   ReportUnused<TextureParams::value_type>( out ) );
}


void Parameters::clear()
{
    m_float_params.clear();
    m_int_params.clear();
    m_vector2_params.clear();
    m_vector3_params.clear();
    m_vector4_params.clear();
    m_color_params.clear();
    m_matrix_params.clear();
    m_string_params.clear();
    m_texture_params.clear();
}
