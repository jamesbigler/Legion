
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

#ifndef LEGION_COMMON_MATH_MATRIX_H_
#define LEGION_COMMON_MATH_MATRIX_H_

#include <Legion/Common/Math/Vector.hpp>


#define MATRIX_ACCESS(m,i,j) m[i*4+j]

namespace legion
{

class Matrix;

Matrix& operator*=( Matrix& m1, const Matrix& m2 );
Matrix& operator-=( Matrix& m1, const Matrix& m2 );
Matrix& operator+=( Matrix& m1, const Matrix& m2 );
Matrix& operator*=( Matrix& m1, float f );
Matrix& operator/=( Matrix& m1, float f );

Matrix  operator-( const Matrix& m1, const Matrix& m2 );
Matrix  operator+( const Matrix& m1, const Matrix& m2 );
Matrix  operator*( const Matrix& m1, const Matrix& m2 );
Vector4 operator*( const Matrix& m, const Vector4& v );
Vector4 operator*( const Vector4& v, const Matrix& m );
Matrix  operator/( const Matrix& m, float f );
Matrix  operator*( const Matrix& m, float f );
Matrix  operator*( float f, const Matrix& m );


class Matrix
{
public:
                  Matrix();

    explicit      Matrix( const float data[16] );

                  Matrix( const Matrix& m );

    Matrix&       operator=( const Matrix& b );

    float         operator[]( unsigned i )const;

    float&        operator[]( unsigned i );

    Vector4       getRow( unsigned m )const;

    Vector4       getCol( unsigned n )const;

    float*        getData();

    const float*  getData()const;

    void          setRow( unsigned m, const Vector4 &r );

    void          setCol( unsigned n, const Vector4 &c );

    Matrix        transpose()const;

    Matrix        inverse()const;

    Vector3       transformPoint( const Vector3& p )const;
    Vector3       transformVector( const Vector3& v )const;

    static Matrix rotate( float radians, const Vector3& axis );

    static Matrix translate( const Vector3& vec );

    static Matrix scale( const Vector3& vec );
    
    static Matrix lookAt( const Vector3& eye,
                          const Vector3& at,
                          const Vector3& up );

    static Matrix identity();


    // Ordered comparison operator so Matrix can be used in an STL container.
    bool          operator<( const Matrix& rhs ) const;

private:
    float         det() const;

    float m_data[16]; // The data array is stored in row-major order.
};


inline Matrix::Matrix()
{
}


inline Matrix::Matrix( const float data[16] ) 
{ 
    for(unsigned int i = 0; i < 16; ++i)
        m_data[i] = data[i];
}


inline Matrix::Matrix( const Matrix& m )
{
    for(unsigned int i = 0; i < 16; ++i)
        m_data[i] = m.m_data[i];
}


inline Matrix& Matrix::operator=( const Matrix& b )
{
    for(unsigned int i = 0; i < 16; ++i)
        m_data[i] = b.m_data[i];
    return *this;
}


inline float Matrix::operator[]( unsigned i )const
{ 
    return m_data[i];
}


inline float& Matrix::operator[]( unsigned i )
{
    return m_data[i];
}


inline Vector4 Matrix::getRow( unsigned int m )const
{
    Vector4 temp;
    const float* row = &( m_data[m*4] );
    temp[0] = row[0];
    temp[1] = row[1];
    temp[2] = row[2];
    temp[3] = row[3];
    return temp;
}


inline Vector4 Matrix::getCol( unsigned int n )const
{
    Vector4 temp;
    temp[0] = MATRIX_ACCESS( m_data, 0, n );
    temp[1] = MATRIX_ACCESS( m_data, 1, n );
    temp[2] = MATRIX_ACCESS( m_data, 2, n );
    temp[3] = MATRIX_ACCESS( m_data, 3, n );
    return temp;
}


inline float* Matrix::getData()
{
    return m_data;
}


inline const float* Matrix::getData() const
{
    return m_data;
}


inline void Matrix::setRow( unsigned int m, const Vector4 &v )
{
    float* row = &( m_data[m*4] );
    row[0] = v[0];
    row[1] = v[1];
    row[2] = v[2];
    row[3] = v[3];
}


inline void Matrix::setCol( unsigned int n, const Vector4 &v )
{
    MATRIX_ACCESS( m_data, 0, n ) = v[0];
    MATRIX_ACCESS( m_data, 1, n ) = v[1];
    MATRIX_ACCESS( m_data, 2, n ) = v[2];
    MATRIX_ACCESS( m_data, 3, n ) = v[3];
}


inline Matrix operator-( const Matrix& m1, const Matrix& m2 )
{
    Matrix temp( m1 );
    temp -= m2;
    return temp;
}


inline Matrix& operator-=( Matrix& m1, const Matrix& m2 )
{
    for ( unsigned int i = 0; i < 16; ++i )
        m1[i] -= m2[i];
    return m1;
}


inline Matrix operator+( const Matrix& m1, const Matrix& m2 )
{
    Matrix temp( m1 );
    temp += m2;
    return temp;
}


inline Matrix& operator+=( Matrix& m1, const Matrix& m2 )
{
    for ( unsigned int i = 0; i < 16; ++i )
        m1[i] += m2[i];
    return m1;
}


inline Matrix operator*( const Matrix& m1, const Matrix& m2 )
{
    Matrix temp;

    for ( unsigned int i = 0; i < 4; ++i ) {
        for ( unsigned int j = 0; j < 4; ++j ) {
            float sum = 0.0f;
            for ( unsigned int k = 0; k < 4; ++k ) {
                float ik = m1[ i*4+k ];
                float kj = m2[ k*4+j ];
                sum += ik * kj;
            }
            temp[i*4+j] = sum;
        }
    }
    return temp;
}


inline Matrix& operator*=( Matrix& m1, const Matrix& m2 )
{
    m1 = m1*m2;
    return m1;
}


inline Vector4 operator*( const Matrix& m, const Vector4& vec )
{
    Vector4 temp;
    temp.setX( m[ 0] * vec.x() +
               m[ 1] * vec.y() +
               m[ 2] * vec.z() +
               m[ 3] * vec.w() );

    temp.setY( m[ 4] * vec.x() +
               m[ 5] * vec.y() +
               m[ 6] * vec.z() +
               m[ 7] * vec.w() );

    temp.setZ( m[ 8] * vec.x() +
               m[ 9] * vec.y() +
               m[10] * vec.z() +
               m[11] * vec.w() );

    temp.setW( m[12] * vec.x() +
               m[13] * vec.y() +
               m[14] * vec.z() +
               m[15] * vec.w() );
    return temp;
}


inline Vector4 operator*( const Vector4& v, const Matrix& m )
{
    Vector4 t;
    for ( unsigned int i = 0; i < 4; ++i )
    {
        float sum = 0.0f;
        for ( unsigned int j = 0; j < 4; ++j )
            sum += v[j] * MATRIX_ACCESS( m, j, i ) ;
        t[i] = sum;
    }
    return t;
}


inline Matrix operator*( const Matrix& m, float f )
{
    Matrix temp( m );
    temp *= f;
    return temp;
}


inline Matrix& operator*=( Matrix& m, float f )
{
    for ( unsigned int i = 0; i < 16; ++i )
        m[i] *= f;
    return m;
}


inline Matrix operator*( float f, const Matrix& m )
{
    Matrix temp;
    for ( unsigned int i = 0; i < 16; ++i )
        temp[i] = m[i]*f;
    return temp;
}


inline Matrix operator/( const Matrix& m, float f )
{
    Matrix temp( m );
    temp /= f;
    return temp;
}


inline Matrix& operator/=( Matrix& m, float f )
{
    float inv_f = 1.0f / f;
    for ( unsigned int i = 0; i < 16; ++i )
        m[i] *= inv_f;
    return m;
}


inline Matrix Matrix::transpose()const
{
    Matrix ret;
    for( unsigned int row = 0; row < 4; ++row )
        for( unsigned int col = 0; col < 4; ++col )
            ret[col*4+row] = m_data[row*4+col];
    return ret;
}


inline float Matrix::det()const
{
    const float* m = m_data;
    float d =
        m[0]*m[5]*m[10]*m[15]-
        m[0]*m[5]*m[11]*m[14]+m[0]*m[9]*m[14]*m[7]-
        m[0]*m[9]*m[6]*m[15]+m[0]*m[13]*m[6]*m[11]-
        m[0]*m[13]*m[10]*m[7]-m[4]*m[1]*m[10]*m[15]+m[4]*m[1]*m[11]*m[14]-
        m[4]*m[9]*m[14]*m[3]+m[4]*m[9]*m[2]*m[15]-
        m[4]*m[13]*m[2]*m[11]+m[4]*m[13]*m[10]*m[3]+m[8]*m[1]*m[6]*m[15]-
        m[8]*m[1]*m[14]*m[7]+m[8]*m[5]*m[14]*m[3]-
        m[8]*m[5]*m[2]*m[15]+m[8]*m[13]*m[2]*m[7]-
        m[8]*m[13]*m[6]*m[3]-
        m[12]*m[1]*m[6]*m[11]+m[12]*m[1]*m[10]*m[7]-
        m[12]*m[5]*m[10]*m[3]+m[12]*m[5]*m[2]*m[11]-
        m[12]*m[9]*m[2]*m[7]+m[12]*m[9]*m[6]*m[3];
    return d;
}


inline Matrix Matrix::inverse()const
{
    Matrix dst;
    const float* m = m_data;
    const float d = 1.0f / det();

    dst[0]  = d * (m[ 5] * (m[10] * m[15] - m[14] * m[11]) +
                   m[ 9] * (m[14] * m[ 7] - m[ 6] * m[15]) +
                   m[13] * (m[ 6] * m[11] - m[10] * m[ 7]));
    dst[4]  = d * (m[ 6] * (m[ 8] * m[15] - m[12] * m[11]) +
                   m[10] * (m[12] * m[ 7] - m[ 4] * m[15]) +
                   m[14] * (m[ 4] * m[11] - m[ 8] * m[ 7]));
    dst[8]  = d * (m[ 7] * (m[ 8] * m[13] - m[12] * m[ 9]) + 
                   m[11] * (m[12] * m[ 5] - m[ 4] * m[13]) +
                   m[15] * (m[ 4] * m[ 9] - m[ 8] * m[ 5]));
    dst[12] = d * (m[ 4] * (m[13] * m[10] - m[ 9] * m[14]) +
                   m[ 8] * (m[ 5] * m[14] - m[13] * m[ 6]) +
                   m[12] * (m[ 9] * m[ 6] - m[ 5] * m[10]));
    dst[1]  = d * (m[ 9] * (m[ 2] * m[15] - m[14] * m[ 3]) +
                   m[13] * (m[10] * m[ 3] - m[ 2] * m[11]) +
                   m[ 1] * (m[14] * m[11] - m[10] * m[15]));
    dst[5]  = d * (m[10] * (m[ 0] * m[15] - m[12] * m[ 3]) +
                   m[14] * (m[ 8] * m[ 3] - m[ 0] * m[11]) +
                   m[ 2] * (m[12] * m[11] - m[ 8] * m[15]));
    dst[9]  = d * (m[11] * (m[ 0] * m[13] - m[12] * m[ 1]) +
                   m[15] * (m[ 8] * m[ 1] - m[ 0] * m[ 9]) +
                   m[ 3] * (m[12] * m[ 9] - m[ 8] * m[13]));
    dst[13] = d * (m[ 8] * (m[13] * m[ 2] - m[ 1] * m[14]) +
                   m[12] * (m[ 1] * m[10] - m[ 9] * m[ 2]) +
                   m[ 0] * (m[ 9] * m[14] - m[13] * m[10]));
    dst[2]  = d * (m[13] * (m[ 2] * m[ 7] - m[ 6] * m[ 3]) +
                   m[ 1] * (m[ 6] * m[15] - m[14] * m[ 7]) +
                   m[ 5] * (m[14] * m[ 3] - m[ 2] * m[15]));
    dst[6]  = d * (m[14] * (m[ 0] * m[ 7] - m[ 4] * m[ 3]) +
                   m[ 2] * (m[ 4] * m[15] - m[12] * m[ 7]) +
                   m[ 6] * (m[12] * m[ 3] - m[ 0] * m[15]));
    dst[10] = d * (m[15] * (m[ 0] * m[ 5] - m[ 4] * m[ 1]) +
                   m[ 3] * (m[ 4] * m[13] - m[12] * m[ 5]) +
                   m[ 7] * (m[12] * m[ 1] - m[ 0] * m[13]));
    dst[14] = d * (m[12] * (m[ 5] * m[ 2] - m[ 1] * m[ 6]) +
                   m[ 0] * (m[13] * m[ 6] - m[ 5] * m[14]) +
                   m[ 4] * (m[ 1] * m[14] - m[13] * m[ 2]));
    dst[3]  = d * (m[ 1] * (m[10] * m[ 7] - m[ 6] * m[11]) +
                   m[ 5] * (m[ 2] * m[11] - m[10] * m[ 3]) +
                   m[ 9] * (m[ 6] * m[ 3] - m[ 2] * m[ 7]));
    dst[7]  = d * (m[ 2] * (m[ 8] * m[ 7] - m[ 4] * m[11]) +
                   m[ 6] * (m[ 0] * m[11] - m[ 8] * m[ 3]) +
                   m[10] * (m[ 4] * m[ 3] - m[ 0] * m[ 7]));
    dst[11] = d * (m[ 3] * (m[ 8] * m[ 5] - m[ 4] * m[ 9]) +
                   m[ 7] * (m[ 0] * m[ 9] - m[ 8] * m[ 1]) +
                   m[11] * (m[ 4] * m[ 1] - m[ 0] * m[ 5]));
    dst[15] = d * (m[ 0] * (m[ 5] * m[10] - m[ 9] * m[ 6]) +
                   m[ 4] * (m[ 9] * m[ 2] - m[ 1] * m[10]) +
                   m[ 8] * (m[ 1] * m[ 6] - m[ 5] * m[ 2]));
    return dst;
}


inline Vector3 Matrix::transformPoint( const Vector3& p )const
{
    Vector4 p4( p, 1.0f );
    p4 = *this * p4;
    return Vector3( p4.x(), p4.y(), p4.z() );
}


inline Vector3 Matrix::transformVector( const Vector3& v )const
{
    Vector4 v4( v, 0.0f );
    v4 = *this * v4;
    return Vector3( v4.x(), v4.y(), v4.z() );
}


inline Matrix Matrix::rotate( float radians, const Vector3& axis )
{
    Matrix m = Matrix::identity();

    // NOTE: Element 0,1 is wrong in Foley and Van Dam, Pg 227!
    const float sintheta = sinf( radians );
    const float costheta = cosf( radians );
    const float ux = axis.x();
    const float uy = axis.y();
    const float uz = axis.z();
    m[0*4+0] = ux*ux+costheta*(1.0f-ux*ux);
    m[0*4+1] = ux*uy*(1.0f-costheta)-uz*sintheta;
    m[0*4+2] = uz*ux*(1.0f-costheta)+uy*sintheta;
    m[0*4+3] = 0.0f;

    m[1*4+0] = ux*uy*(1-costheta)+uz*sintheta;
    m[1*4+1] = uy*uy+costheta*(1.0f-uy*uy);
    m[1*4+2] = uy*uz*(1-costheta)-ux*sintheta;
    m[1*4+3] = 0.0f;

    m[2*4+0] = uz*ux*(1.0f-costheta)-uy*sintheta;
    m[2*4+1] = uy*uz*(1.0f-costheta)+ux*sintheta;
    m[2*4+2] = uz*uz+costheta*(1.0f-uz*uz);
    m[2*4+3] = 0.0f;

    m[3*4+0] = 0.0f;
    m[3*4+1] = 0.0f;
    m[3*4+2] = 0.0f;
    m[3*4+3] = 1.0f;

    return  m;
}


inline Matrix Matrix::translate( const Vector3& vec )
{
    Matrix m = Matrix::identity();
    m[3] = vec.x();
    m[7] = vec.y();
    m[11]= vec.z();
    return m;
}


inline Matrix Matrix::scale( const Vector3& vec )
{
    Matrix m = Matrix::identity();
    m[0] = vec.x();
    m[5] = vec.y();
    m[10]= vec.z();
    return m; 
}


inline Matrix Matrix::lookAt( const Vector3& eye,
                              const Vector3& at,
                              const Vector3& up )
{


    // create a viewing matrix
    Vector3 w = normalize( eye - at );
    Vector3 u = normalize( cross( up, w ) );
    Vector3 v = cross( w, u );

    Matrix view = identity();
    view[ 0] = u.x();
    view[ 1] = u.y();
    view[ 2] = u.z();
    view[ 4] = v.x();
    view[ 5] = v.y();
    view[ 6] = v.z();
    view[ 8] = w.x();
    view[ 9] = w.y();
    view[10] = w.z();

    // translate eye to xyz origin
    Matrix move = identity();
    move[ 3] = -(eye.x());
    move[ 7] = -(eye.y());
    move[11] = -(eye.z());

    Matrix ret = ( view * move ).inverse();
    return ret;
}


inline Matrix Matrix::identity()
{
    const float temp[16] = { 1.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 1.0f };
    return Matrix( temp );
}


inline bool Matrix::operator<( const Matrix& rhs )const
{
    for( unsigned int i = 0; i < 16; ++i )
    {
        if( m_data[i] != rhs.m_data[i] ) 
            return m_data[i] < rhs.m_data[i];
    }
    return false;
}


} // namespace legion


#endif // LEGION_COMMON_MATH_MATRIX_H_

#undef MATRIX_ACCESS
