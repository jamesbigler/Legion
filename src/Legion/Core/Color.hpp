
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
    Color( float r, float g, float b, float a );

    float red()const               { return m_c[0]; }
    float green()const             { return m_c[1]; }
    float blue()const              { return m_c[2]; }
    float alpha()const             { return m_c[2]; }

    void  setRed  ( float r )      { m_c[0] = r; }
    void  setGreen( float g )      { m_c[1] = g; }
    void  setBlue ( float b )      { m_c[2] = b; }
    void  setAlpha( float a )      { m_c[3] = a; }

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
private:

    float m_c[4];
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
    m_c[3] = 1.0f;
}


inline Color::Color( float r, float g, float b )
{
    m_c[0] = r;
    m_c[1] = g;
    m_c[2] = b;
    m_c[3] = 1.0f;
}

inline Color::Color( float r, float g, float b, float a )
{
    m_c[0] = r;
    m_c[1] = g;
    m_c[2] = b;
    m_c[3] = a;
}

inline Color& Color::operator=( const Color& rhs )
{
    m_c[0] = rhs.m_c[0];
    m_c[1] = rhs.m_c[1];
    m_c[2] = rhs.m_c[2];
    m_c[3] = rhs.m_c[3];
    return *this;
}

inline Color& Color::operator*=( const Color& rhs )
{
    m_c[0] *= rhs.m_c[0];
    m_c[1] *= rhs.m_c[1];
    m_c[2] *= rhs.m_c[2];
    m_c[3] *= rhs.m_c[3];
    return *this;
}

inline Color& Color::operator/=( const Color& rhs )
{
    m_c[0] /= rhs.m_c[0];
    m_c[1] /= rhs.m_c[1];
    m_c[2] /= rhs.m_c[2];
    m_c[3] /= rhs.m_c[3];
    return *this;
}

inline Color& Color::operator+=( const Color& rhs )
{
    m_c[0] += rhs.m_c[0];
    m_c[1] += rhs.m_c[1];
    m_c[2] += rhs.m_c[2];
    m_c[3] += rhs.m_c[3];
    return *this;
}

inline Color& Color::operator-=( const Color& rhs )
{
    m_c[0] -= rhs.m_c[0];
    m_c[1] -= rhs.m_c[1];
    m_c[2] -= rhs.m_c[2];
    m_c[3] -= rhs.m_c[3];
    return *this;
}

inline Color& Color::operator*=( float rhs )
{
    m_c[0] *= rhs;
    m_c[1] *= rhs;
    m_c[2] *= rhs;
    m_c[3] *= rhs;
    return *this;
}

inline Color& Color::operator/=( float rhs )
{
    m_c[0] /= rhs;
    m_c[1] /= rhs;
    m_c[2] /= rhs;
    m_c[3] /= rhs;
    return *this;
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
