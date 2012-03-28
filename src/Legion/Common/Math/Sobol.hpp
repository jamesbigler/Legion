

#ifndef LEGION_COMMON_MATH_SOBOL_HPP_
#define LEGION_COMMON_MATH_SOBOL_HPP_

#include <Legion/Common/Math/Vector.hpp>

namespace legion
{


class Sobol
{
public:
    enum Dimension
    {
        DIM_PIXEL_X = 0, // Pixel dims must be 0,1
        DIM_PIXEL_Y,
        DIM_BSDF_X,
        DIM_BSDF_Y,
        DIM_LENS_X,
        DIM_LENS_Y,
        DIM_TIME,
        DIM_SHADOW_X,
        DIM_SHADOW_Y
    };
    
    /// Generate an single element A_i,dim of the Sobol sequence which is the
    /// dimth dimension of the ith sobol vector. Scrambled by s. 
    static float    gen ( unsigned i, unsigned dim,  unsigned scramble = 0u);
    static unsigned genu( unsigned i, unsigned dim,  unsigned scramble = 0u);


    static const unsigned MAX_DIMS = 256u;
    static const unsigned MATRICES[ MAX_DIMS*52u ];
private:
    static float intAsFloat( int x );

    Sobol();
};



inline unsigned Sobol::genu( unsigned i, unsigned dim,  unsigned s )
{
    const unsigned int adr = dim*(52/4);
    unsigned int result = 0;

    for (unsigned int c = 0; (i != 0); i>>=8, c+=2) {

        const Index4 matrix1 = reinterpret_cast<const Index4*>( MATRICES )[adr + c + 0];
        const Index4 matrix2 = reinterpret_cast<const Index4*>( MATRICES )[adr + c + 1];

        result ^= (((matrix1.x()&(unsigned)(-((int) i     & 1)))   ^
                    (matrix1.y()&(unsigned)(-((int)(i>>1) & 1))))  ^
                   ((matrix1.z()&(unsigned)(-((int)(i>>2) & 1)))   ^
                    (matrix1.w()&(unsigned)(-((int)(i>>3) & 1))))) ^
                  (((matrix2.x()&(unsigned)(-((int)(i>>4) & 1)))   ^
                    (matrix2.y()&(unsigned)(-((int)(i>>5) & 1))))  ^
                   ((matrix2.z()&(unsigned)(-((int)(i>>6) & 1)))   ^
                    (matrix2.w()&(unsigned)(-((int)(i>>7) & 1)))));
    }

    for (unsigned c = 8; ((s != 0) && (c < 13)); s>>=4, ++c) {
        const Index4 matrix1 = reinterpret_cast<const Index4*>( MATRICES )[adr + c];
        result ^= (((matrix1.x()&(unsigned)(-((int) s     & 1)))  ^
                    (matrix1.y()&(unsigned)(-((int)(s>>1) & 1)))) ^
                   ((matrix1.z()&(unsigned)(-((int)(s>>2) & 1)))  ^
                    (matrix1.w()&(unsigned)(-((int)(s>>3) & 1)))));
    }

    return result;
}


inline float Sobol::gen( unsigned i, unsigned dim,  unsigned scramble )
{
    return intAsFloat( 0x3F800000 | ( genu( i, dim, scramble ) >> 9 ) ) - 1.0f;
}


inline float Sobol::intAsFloat( int x )
{
  return *(float*)&x;
}


}

#endif //  LEGION_COMMON_MATH_SOBOL_HPP_
