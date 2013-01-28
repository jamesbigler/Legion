
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

#ifndef LEGION_COMMON_MATH_SOBOL_HPP_
#define LEGION_COMMON_MATH_SOBOL_HPP_

#include <Legion/Common/Math/CUDA/sobol_matrices.hpp>
#include <Legion/Common/Math/CUDA/Math.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>

namespace legion
{


class Sobol
{
public:

    /// Generate a single element A_i,dim of the Sobol sequence which is the
    /// dimth dimension of the ith sobol vector. Scrambled by s. 
    LDEVICE static float    gen ( uint64 i, unsigned dim, unsigned s = 0u );
    LDEVICE static unsigned genu( uint64 i, unsigned dim, unsigned s = 0u );

    // 2^m should be > film max_dim
    LDEVICE static void getRasterPos( const unsigned m,
                                      const unsigned pass,
                                      const uint2&   pixel,
                                      float2&        raster_pos,
                                      uint64&        sobol_index );

private:
    LDEVICE static uint64 lookUpSobolIndex( const unsigned m,
                                            const unsigned pass,
                                            const uint2&   pixel );
    LDEVICE Sobol();
};


LDEVICE inline float Sobol::gen( uint64 i, unsigned dim,  unsigned scramble )
{
    return __int_as_float( 0x3F800000 | (genu( i, dim, scramble ) >> 9 ))-1.0f;
}


LDEVICE inline unsigned Sobol::genu( uint64 i, unsigned dim,  unsigned s )
{
    const unsigned int adr = dim*(52/4);
    unsigned int result = 0;

    for (unsigned int c = 0; (i != 0); i>>=8, c+=2) {

        const uint4 matrix1 =
            reinterpret_cast<const uint4*>( SOBOL_MATRICES )[adr + c + 0];
        const uint4 matrix2 =
            reinterpret_cast<const uint4*>( SOBOL_MATRICES )[adr + c + 1];

        result ^= (((matrix1.x&(unsigned)(-((int) i     & 1)))   ^
                    (matrix1.y&(unsigned)(-((int)(i>>1) & 1))))  ^
                   ((matrix1.z&(unsigned)(-((int)(i>>2) & 1)))   ^
                    (matrix1.w&(unsigned)(-((int)(i>>3) & 1))))) ^
                  (((matrix2.x&(unsigned)(-((int)(i>>4) & 1)))   ^
                    (matrix2.y&(unsigned)(-((int)(i>>5) & 1))))  ^
                   ((matrix2.z&(unsigned)(-((int)(i>>6) & 1)))   ^
                    (matrix2.w&(unsigned)(-((int)(i>>7) & 1)))));
    }

    for (unsigned c = 8; ((s != 0) && (c < 13)); s>>=4, ++c) {
        const uint4 matrix1 =
            reinterpret_cast<const uint4*>( SOBOL_MATRICES )[adr + c];
        result ^= (((matrix1.x&(unsigned)(-((int) s     & 1)))  ^
                    (matrix1.y&(unsigned)(-((int)(s>>1) & 1)))) ^
                   ((matrix1.z&(unsigned)(-((int)(s>>2) & 1)))  ^
                    (matrix1.w&(unsigned)(-((int)(s>>3) & 1)))));
    }

    return result;
}



// This function courtesy of Alex Keller, based on 'Enumerating Quasi-Monte
// Carlo Point Sequences in Elementary Intervals', Gruenschloß, Raab, and
// Keller
LDEVICE inline uint64 Sobol::lookUpSobolIndex( const unsigned m,
                                               const unsigned pass,
                                               const uint2&  pixel )
{
    
    const uint32 m2 = m << 1;
    uint32 frame = pass;
    uint64 index = uint64(frame) << m2;

    // the delta value only depends on frame and m, thus it can be cached
    // across multiple function calls, if desired.
    uint64 delta = 0;
    for (uint32 c = 0; frame; frame >>= 1, ++c)
        if (frame & 1) // Add flipped column m + c + 1.
            delta ^= vdc_sobol_matrices[m - 1][c];

    // flipped b
    uint64 b = ( ( static_cast<uint64>(pixel.x) << m ) | pixel.y ) ^ delta;

    for (uint32 c = 0; b; b >>= 1, ++c)
        if (b & 1) // Add column 2 * m - c.
            index ^= vdc_sobol_matrices_inv[m - 1][c];

    return index;
}

     
    
// 2^m should be > film max_dim
LDEVICE inline void Sobol::getRasterPos( const unsigned m,
                                         const unsigned pass,
                                         const uint2&   pixel,
                                         float2&        raster_pos,
                                         uint64& sobol_index )
{
  sobol_index = lookUpSobolIndex( m, pass, pixel );
  raster_pos.x = static_cast<float>( Sobol::genu( sobol_index, 0 ) ) /
                                     ( 1U << (32-m) );
  raster_pos.y = static_cast<float>( Sobol::genu( sobol_index, 1 ) ) /
                                     ( 1U << (32-m) );
}

}

#endif //  LEGION_COMMON_MATH_SOBOL_HPP_
