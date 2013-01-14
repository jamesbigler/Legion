
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

#include <Legion/Common/Math/sobol_matrices.hpp>

namespace legion
{


class Sobol
{
public:
    /// Generate a single element A_i,dim of the Sobol sequence which is the
    /// dimth dimension of the ith sobol vector. Scrambled by s. 
    __device__ static float    gen ( unsigned i, unsigned dim,  unsigned scramble = 0u );
    __device__ static unsigned genu( unsigned i, unsigned dim,  unsigned scramble = 0u );

    __device__ static float2   genPixelSample( unsigned i, unsigned scramble = 0u );
    __device__ static float2   genLensSample ( unsigned i, unsigned scramble = 0u );
    __device__ static float    genTimeSample ( unsigned i, unsigned scramble = 0u );

    __device__ static void getRasterPos( const unsigned m, // 2^m should be > film max_dim
                                         const unsigned pass,
                                         const uint2&   pixel,
                                         float2&        raster_pos,
                                         unsigned&      sobol_index );

    //__device__ static const unsigned MAX_DIMS = 256u;
    //__device__ static const unsigned MATRICES[ MAX_DIMS*52u ];
private:
    __device__ static unsigned lookUpSobolIndex( const unsigned m,
                                                 const unsigned pass,
                                                 const uint2&   pixel );
    //static float intAsFloat( int x );

    Sobol();
};


__device__ __inline__ float Sobol::gen( unsigned i, unsigned dim,  unsigned scramble )
{
    //return intAsFloat( 0x3F800000 | ( genu( i, dim, scramble ) >> 9 ) ) - 1.0f;
    return __int_as_float( 0x3F800000 | ( genu( i, dim, scramble ) >> 9 ) ) - 1.0f;
}


__device__ __inline__ float2 Sobol::genPixelSample( unsigned i, unsigned scramble )
{ 
    return make_float2( Sobol::gen( i, 0, scramble ),
                        Sobol::gen( i, 1, scramble ) );
}


__device__ __inline__ float2 Sobol::genLensSample ( unsigned i, unsigned scramble )
{ 
    return make_float2( Sobol::gen( i, 2, scramble ),
                        Sobol::gen( i, 3, scramble ) );
}

__device__ __inline__ float  Sobol::genTimeSample ( unsigned i, unsigned scramble )
{ 
    return Sobol::gen( i, 4, scramble );
}




//inline float Sobol::intAsFloat( int x )
//{
//  return *(float*)&x;
//}

__device__ __inline__ unsigned Sobol::genu( unsigned i, unsigned dim,  unsigned s )
{
    const unsigned int adr = dim*(52/4);
    unsigned int result = 0;

    for (unsigned int c = 0; (i != 0); i>>=8, c+=2) {

        const uint4 matrix1 = reinterpret_cast<const uint4*>( SOBOL_MATRICES )[adr + c + 0];
        const uint4 matrix2 = reinterpret_cast<const uint4*>( SOBOL_MATRICES )[adr + c + 1];
        /*
        const uint4 matrix1 = make_uint4( SOBOL_MATRICES[( adr + c + 0)*4 + 0 ],
                                          SOBOL_MATRICES[( adr + c + 0)*4 + 1 ],
                                          SOBOL_MATRICES[( adr + c + 0)*4 + 2 ],
                                          SOBOL_MATRICES[( adr + c + 0)*4 + 3 ] );
        const uint4 matrix2 = make_uint4( SOBOL_MATRICES[( adr + c + 1)*4 + 0 ],
                                          SOBOL_MATRICES[( adr + c + 1)*4 + 1 ],
                                          SOBOL_MATRICES[( adr + c + 1)*4 + 2 ],
                                          SOBOL_MATRICES[( adr + c + 1)*4 + 3 ] );
                                          */
        /*
        const uint4 matrix1 = make_uint4( SOBOL_MATRICES[(  0)*4 + 0 ],
                                          SOBOL_MATRICES[(  0)*4 + 1 ],
                                          SOBOL_MATRICES[(  0)*4 + 2 ],
                                          SOBOL_MATRICES[(  0)*4 + 3 ] );
        const uint4 matrix2 = make_uint4( SOBOL_MATRICES[(  1)*4 + 0 ],
                                          SOBOL_MATRICES[(  1)*4 + 1 ],
                                          SOBOL_MATRICES[(  1)*4 + 2 ],
                                          SOBOL_MATRICES[(  1)*4 + 3 ] );
                                          */

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
        const uint4 matrix1 = reinterpret_cast<const uint4*>( SOBOL_MATRICES )[adr + c];
        /*
        const uint4 matrix1 = make_uint4( SOBOL_MATRICES[(adr + c)*4 + 0],
                                          SOBOL_MATRICES[(adr + c)*4 + 1],
                                          SOBOL_MATRICES[(adr + c)*4 + 2],
                                          SOBOL_MATRICES[(adr + c)*4 + 3] );
                                          */
        /*
        const uint4 matrix1 = make_uint4( SOBOL_MATRICES[(0)*4 + 0],
                                          SOBOL_MATRICES[(0)*4 + 1],
                                          SOBOL_MATRICES[(0)*4 + 2],
                                          SOBOL_MATRICES[(0)*4 + 3] );
                                          */
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
__device__ __inline__  unsigned Sobol::lookUpSobolIndex( const unsigned m,
                                                         const unsigned pass,
                                                         const uint2&  pixel )
{
    typedef unsigned int       uint32;
    typedef unsigned long long uint64;
    
    const uint32 m2 = m << 1;
    uint32 frame = pass;
    uint64 index = uint64(frame) << m2;

    // TODO: the delta value only depends on frame
    // and m, thus it can be cached across multiple
    // function calls, if desired.
    uint64 delta = 0;
    for (uint32 c = 0; frame; frame >>= 1, ++c)
        if (frame & 1) // Add flipped column m + c + 1.
            delta ^= vdc_sobol_matrices[m - 1][c];

    uint64 b = ( ( static_cast<uint64>(pixel.x) << m ) | pixel.y ) ^ delta; // flipped b

    for (uint32 c = 0; b; b >>= 1, ++c)
        if (b & 1) // Add column 2 * m - c.
            index ^= vdc_sobol_matrices_inv[m - 1][c];

    return index;
}

     
    
__device__ __inline__ void Sobol::getRasterPos( const unsigned m, // 2^m should be > film max_dim
                                                const unsigned pass,
                                                const uint2&   pixel,
                                                float2&        raster_pos,
                                                unsigned&      sobol_index )
{
  sobol_index = lookUpSobolIndex( m, pass, pixel );
  raster_pos.x = static_cast<float>(Sobol::genu( sobol_index, 0 ) ) / (1U<<(32-m) );
  raster_pos.y = static_cast<float>(Sobol::genu( sobol_index, 1 ) ) / (1U<<(32-m) );
}

}

#endif //  LEGION_COMMON_MATH_SOBOL_HPP_
