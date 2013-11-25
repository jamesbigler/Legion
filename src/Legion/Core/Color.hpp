
#ifndef LEGION_CORE_COLOR_HPP_
#define LEGION_CORE_COLOR_HPP_

#include <Legion/Common/Util/Preprocessor.hpp>

namespace legion
{


class LAPI Color
{
public:
    LAPI Color() {}
    LAPI explicit Color( float rgb );
    LAPI Color( float r, float g, float b );

    LAPI float red()const               { return m_c[0]; }
    LAPI float green()const             { return m_c[1]; }
    LAPI float blue()const              { return m_c[2]; }
    LAPI 
    LAPI float r()const                 { return m_c[0]; }
    LAPI float g()const                 { return m_c[1]; }
    LAPI float b()const                 { return m_c[2]; }

    LAPI void  setRed  ( float r )      { m_c[0] = r; }
    LAPI void  setGreen( float g )      { m_c[1] = g; }
    LAPI void  setBlue ( float b )      { m_c[2] = b; }

    LAPI float  operator[]( unsigned idx )const { return m_c[idx]; }
    LAPI float& operator[]( unsigned idx )      { return m_c[idx]; }

    LAPI Color& operator= ( const Color& rhs );
    LAPI Color& operator*=( const Color& rhs );
    LAPI Color& operator/=( const Color& rhs );
    LAPI Color& operator+=( const Color& rhs );
    LAPI Color& operator-=( const Color& rhs );
    LAPI Color& operator*=( float rhs );
    LAPI Color& operator/=( float rhs );

    LAPI float luminance()const;
    LAPI float sum()const;
private:

    float m_c[3];
};


Color operator*( const Color& c0, const Color& c1 );
Color operator/( const Color& c0, const Color& c1 );
Color operator+( const Color& c0, const Color& c1 );
Color operator-( const Color& c0, const Color& c1 );
Color operator*( float f, const Color& c );
Color operator*( const Color& c, float f );
Color operator/( const Color& c, float f );

    
inline Color::Color( float rgb )
{
    m_c[0] = m_c[1] = m_c[2] = rgb;
}


inline Color::Color( float r, float g, float b )
{
    m_c[0] = r;
    m_c[1] = g;
    m_c[2] = b;
}


inline Color& Color::operator=( const Color& rhs )
{
    m_c[0] = rhs.m_c[0];
    m_c[1] = rhs.m_c[1];
    m_c[2] = rhs.m_c[2];
    return *this;
}

inline Color& Color::operator*=( const Color& rhs )
{
    m_c[0] *= rhs.m_c[0];
    m_c[1] *= rhs.m_c[1];
    m_c[2] *= rhs.m_c[2];
    return *this;
}

inline Color& Color::operator/=( const Color& rhs )
{
    m_c[0] /= rhs.m_c[0];
    m_c[1] /= rhs.m_c[1];
    m_c[2] /= rhs.m_c[2];
    return *this;
}

inline Color& Color::operator+=( const Color& rhs )
{
    m_c[0] += rhs.m_c[0];
    m_c[1] += rhs.m_c[1];
    m_c[2] += rhs.m_c[2];
    return *this;
}

inline Color& Color::operator-=( const Color& rhs )
{
    m_c[0] -= rhs.m_c[0];
    m_c[1] -= rhs.m_c[1];
    m_c[2] -= rhs.m_c[2];
    return *this;
}

inline Color& Color::operator*=( float rhs )
{
    m_c[0] *= rhs;
    m_c[1] *= rhs;
    m_c[2] *= rhs;
    return *this;
}

inline Color& Color::operator/=( float rhs )
{
    m_c[0] /= rhs;
    m_c[1] /= rhs;
    m_c[2] /= rhs;
    return *this;
}

inline float Color::luminance()const
{
    return 0.2126f*m_c[0] + 0.7152f*m_c[1] + 0.0722f*m_c[2]; 
}


inline float Color::sum()const
{
    return m_c[0] + m_c[1] + m_c[2]; 
}



/******************************************************************************\
 *
 *  Free functions
 *
\******************************************************************************/

inline Color operator*( const Color& c0, const Color& c1 )
{
    Color t( c0 );
    t *= c1;
    return t;
}

inline Color operator/( const Color& c0, const Color& c1 )
{
    Color t( c0 );
    t /= c1;
    return t;
}

inline Color operator+( const Color& c0, const Color& c1 )
{
    Color t( c0 );
    t += c1;
    return t;
}

inline Color operator-( const Color& c0, const Color& c1 )
{
    Color t( c0 );
    t -= c1;
    return t;
}

inline Color operator*( float f, const Color& c )
{
    Color t( c );
    t *= f;
    return t;
}

inline Color operator*( const Color& c, float f )
{
    Color t( c );
    t *= f;
    return t;
}

inline Color operator/( const Color& c, float f )
{
    Color t( c );
    t /= f;
    return t;
}


}

#endif //  LEGION_CORE_COLOR_HPP_
