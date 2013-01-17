
#ifndef LEGION_CORE_COLOR_HPP_
#define LEGION_CORE_COLOR_HPP_

namespace legion
{


class Color
{
public:
    Color() {}
    explicit Color( float rgb );
    Color( float r, float g, float b );

    float red()const               { return m_c[0]; }
    float green()const             { return m_c[1]; }
    float blue()const              { return m_c[2]; }
    
    float r()const                 { return m_c[0]; }
    float g()const                 { return m_c[1]; }
    float b()const                 { return m_c[2]; }

    void  setRed  ( float r )      { m_c[0] = r; }
    void  setGreen( float g )      { m_c[1] = g; }
    void  setBlue ( float b )      { m_c[2] = b; }

    float  operator[]( unsigned idx )const;
    float& operator[]( unsigned idx );

    Color& operator= ( const Color& rhs );
    Color& operator*=( const Color& rhs );
    Color& operator/=( const Color& rhs );
    Color& operator+=( const Color& rhs );
    Color& operator-=( const Color& rhs );
    Color& operator*=( float rhs );
    Color& operator/=( float rhs );

    float luminance()const;
    float sum()const;
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
