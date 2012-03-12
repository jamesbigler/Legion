
/// \file Vector.hpp
/// Vector (numeric tuple) class of arbitrary dimension and element type
#ifndef LEGION_CORE_VECTOR_H_
#define LEGION_CORE_VECTOR_H_

#include <cmath>
#include <Legion/Common/Util/Assert.hpp>

namespace legion
{

//template<unsigned DIM, typename TYPE> class Vector;



/// Vector (numeric tuple) class of arbitrary dimension and element type
template <unsigned DIM, typename TYPE>
class Vector
{
public:
    Vector();
    Vector( TYPE t );
    Vector( TYPE x, TYPE y );                      // Only valid for Vector<2>
    Vector( TYPE x, TYPE y, TYPE z );              // Only valid for Vector<3>
    Vector( TYPE x, TYPE y, TYPE z, TYPE w  );     // Only valid for Vector<4>
    Vector( const Vector& v );
    Vector( const Vector<DIM-1,TYPE>& v, TYPE last );
    explicit Vector( const TYPE v[DIM] );
    
    template <typename TYPE2>
    explicit Vector( const Vector<DIM, TYPE2>& v );

    TYPE x() const;
    TYPE y() const;
    TYPE z() const;                                // Only valid for Vector<3,4>
    TYPE w() const;                                // Only valid for Vector<4>
    void setX( TYPE x );
    void setY( TYPE y );
    void setZ( TYPE z );                           // Only valid for Vector<3,4>
    void setW( TYPE w );                           // Only valid for Vector<3>


    TYPE  operator[]( unsigned idx ) const;
    TYPE& operator[]( unsigned idx );

    Vector<DIM, TYPE>& operator= ( const Vector<DIM, TYPE>& rhs );
    Vector<DIM, TYPE>& operator*=( const Vector<DIM, TYPE>& rhs );
    Vector<DIM, TYPE>& operator/=( const Vector<DIM, TYPE>& rhs );
    Vector<DIM, TYPE>& operator+=( const Vector<DIM, TYPE>& rhs );
    Vector<DIM, TYPE>& operator-=( const Vector<DIM, TYPE>& rhs );
    Vector<DIM, TYPE>& operator*=( TYPE rhs );
    Vector<DIM, TYPE>& operator/=( TYPE rhs );

    TYPE length() const;
    TYPE lengthSquared() const;
    TYPE normalize();
private:
    TYPE m_v[ DIM ];
};


#define TEMPLATE_DECL template<unsigned DIM, typename TYPE> 
TEMPLATE_DECL Vector<DIM, TYPE> operator*( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 );
TEMPLATE_DECL Vector<DIM, TYPE> operator/( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 );
TEMPLATE_DECL Vector<DIM, TYPE> operator+( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 );
TEMPLATE_DECL Vector<DIM, TYPE> operator-( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 );
TEMPLATE_DECL Vector<DIM, TYPE> operator*( TYPE f, const Vector<DIM, TYPE>& v );
TEMPLATE_DECL Vector<DIM, TYPE> operator*( const Vector<DIM, TYPE>& v, TYPE f );
TEMPLATE_DECL Vector<DIM, TYPE> operator/( const Vector<DIM, TYPE>& v, TYPE f );
TEMPLATE_DECL Vector<DIM, TYPE> operator-( const Vector<DIM, TYPE>& v );
TEMPLATE_DECL Vector<DIM, TYPE> normalize( const Vector<DIM, TYPE>& v );
TEMPLATE_DECL TYPE dot( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 );
#undef TEMPLATE_DECL

template<typename TYPE> Vector<3, TYPE> cross( const Vector<3, TYPE>& v0, const Vector<3, TYPE>& v1 );



// TODO: #include <Private/Vector.impl>


    
/******************************************************************************\
 * 
 * Vector class implementaion 
 *
\******************************************************************************/

template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector() 
{
}

template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector( TYPE t )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] = t;
}

template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector( TYPE x, TYPE y )
{
    LEGION_STATIC_ASSERT( DIM == 2 );
    m_v[0] = x;
    m_v[1] = y;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector( TYPE x, TYPE y, TYPE z )
{
    LEGION_STATIC_ASSERT( DIM == 3 );
    m_v[0] = x;
    m_v[1] = y;
    m_v[2] = z;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector( TYPE x, TYPE y, TYPE z, TYPE w  )
{
    LEGION_STATIC_ASSERT( DIM == 4 );
    m_v[0] = x;
    m_v[1] = y;
    m_v[2] = z;
    m_v[3] = w;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector( const Vector<DIM, TYPE>& v )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] = v[i];
}

template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector( const Vector<DIM-1, TYPE>& v, TYPE last )
{
    for( unsigned i = 0; i < DIM-1; ++i ) m_v[i] = v[i];
    m_v[DIM-1] = last;
}

template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>::Vector( const TYPE v[DIM] )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] = v[i];
}


template<unsigned DIM, typename TYPE>
template<typename TYPE2>
inline Vector<DIM, TYPE>::Vector( const Vector<DIM, TYPE2>& v )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] = v[i];
}


template<unsigned DIM, typename TYPE>
inline TYPE Vector<DIM, TYPE>::x() const
{ 
    return m_v[0];
}


template<unsigned DIM, typename TYPE>
inline TYPE Vector<DIM, TYPE>::y() const
{ 
    return m_v[1];
}


template<unsigned DIM, typename TYPE>
inline TYPE Vector<DIM, TYPE>::z() const
{
    LEGION_STATIC_ASSERT( DIM > 2 );
    return m_v[2];
}


template<unsigned DIM, typename TYPE>
inline TYPE Vector<DIM, TYPE>::w() const
{
    LEGION_STATIC_ASSERT( DIM > 3 );
    return m_v[3];
}


template<unsigned DIM, typename TYPE>
void Vector<DIM, TYPE>::setX( TYPE x )
{
    m_v[0] = x;
}      


template<unsigned DIM, typename TYPE>
void Vector<DIM, TYPE>::setY( TYPE y )
{ 
    m_v[1] = y;
}

template<unsigned DIM, typename TYPE>
void Vector<DIM, TYPE>::setZ( TYPE z )
{
    LEGION_STATIC_ASSERT( DIM > 2 );
    m_v[2] = z;
}


template<unsigned DIM, typename TYPE>
void Vector<DIM, TYPE>::setW( TYPE w )
{
    LEGION_STATIC_ASSERT( DIM > 3 );
    m_v[3] = w;
}


template<unsigned DIM, typename TYPE>
TYPE Vector<DIM, TYPE>::operator[]( unsigned idx ) const
{
    // TODO: legion::Assert( 0 <= idx && idx < DIM );
    return m_v[idx];
}


template<unsigned DIM, typename TYPE>
TYPE& Vector<DIM, TYPE>::operator[]( unsigned idx )
{
    // TODO: legion::Assert( 0 <= idx && idx < DIM );
    return m_v[idx];
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>& Vector<DIM, TYPE>::operator=( const Vector<DIM, TYPE>& rhs )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] = rhs[i];
    return *this;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>& Vector<DIM, TYPE>::operator*=( const Vector<DIM, TYPE>& rhs )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] *= rhs[i];
    return *this;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>& Vector<DIM, TYPE>::operator/=( const Vector<DIM, TYPE>& rhs )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] /= rhs[i];
    return *this;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>& Vector<DIM, TYPE>::operator+=( const Vector<DIM, TYPE>& rhs )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] += rhs[i];
    return *this;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>& Vector<DIM, TYPE>::operator-=( const Vector<DIM, TYPE>& rhs )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] -= rhs[i];
    return *this;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>& Vector<DIM, TYPE>::operator*=( TYPE rhs )
{
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] *= rhs;
    return *this;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE>& Vector<DIM, TYPE>::operator/=( TYPE rhs )
{
    TYPE inv = static_cast<TYPE>( 1.0 ) / rhs;
    for( unsigned i = 0; i < DIM; ++i ) m_v[i] *= inv;
    return *this;
}


template<unsigned DIM, typename TYPE>
inline TYPE Vector<DIM, TYPE>::length() const
{
    return sqrtf( lengthSquared() );
}


template<unsigned DIM, typename TYPE>
inline TYPE Vector<DIM, TYPE>::lengthSquared() const
{
    TYPE len2 = 0.0f;
    for( unsigned i = 0; i < DIM; ++i ) len2 += m_v[i]*m_v[i]; 
    return len2;
}


template<unsigned DIM, typename TYPE>
inline TYPE Vector<DIM, TYPE>::normalize()
{
    // TODO: check out the resulting assembly on this
    TYPE len = length();
    *this /= len;
    return len;
}


/******************************************************************************\
 * 
 * Vector0 specialization
 *
\******************************************************************************/

template<typename TYPE>
class Vector<0, TYPE>
{
private:
    Vector() {}
};

/******************************************************************************\
 * 
 * Free function operators 
 *
\******************************************************************************/

template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator*( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = v0[i] * v1[i];
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator/( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = v0[i] / v1[i];
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator+( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = v0[i] + v1[i];
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator-( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = v0[i] - v1[i];
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator*( TYPE f, const Vector<DIM, TYPE>& v )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = f * v[i];
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator*(  const Vector<DIM, TYPE>& v, TYPE f )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = v[i] * f;
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator/( const Vector<DIM, TYPE>& v, TYPE f )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = v[i] / f;
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> operator-( const Vector<DIM, TYPE>& v )
{
    Vector<DIM, TYPE> result;
    for( unsigned i = 0; i < DIM; ++i ) result[i] = -v[i];
    return result;
}


template<unsigned DIM, typename TYPE>
inline TYPE dot( const Vector<DIM, TYPE>& v0, const Vector<DIM, TYPE>& v1 )
{
    TYPE result = 0;
    for( unsigned i = 0; i < DIM; ++i ) result += v0[i] * v1[i];
    return result;
}


template<unsigned DIM, typename TYPE>
inline Vector<DIM, TYPE> normalize( const Vector<DIM, TYPE>& v )
{
    Vector<DIM, TYPE> temp( v );
    temp.normalize();
    return temp;
}


template<typename TYPE>
inline Vector<3, TYPE> cross( const Vector<3, TYPE>& v0, const Vector<3, TYPE>& v1 )
{
    Vector<3, TYPE> result( v0.y()*v1.z() - v0.z()*v1.y(),
                            v0.z()*v1.x() - v0.x()*v1.z(),
                            v0.x()*v1.y() - v0.y()*v1.x() );
    return result;
}


typedef Vector<2, float> Vector2;
typedef Vector<3, float> Vector3;
typedef Vector<4, float> Vector4;
typedef Vector<2, unsigned> Index2;
typedef Vector<3, unsigned> Index3;
typedef Vector<4, unsigned> Index4;

}

#endif // LEGION_CORE_VECTOR_H_
