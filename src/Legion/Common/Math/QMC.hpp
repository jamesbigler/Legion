
#ifndef LEGION_COMMON_MATH_HPP_
#define LEGION_COMMON_MATH_HPP_

#include <Legion/Common/Math/Vector.hpp>
#include <climits>

namespace legion
{

template<unsigned int N>
inline unsigned int tea( unsigned int val0, unsigned int val1 )
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for( unsigned int n = 0; n < N; n++ )
    {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }

    return v0;
}

// Generate random unsigned int in [0, 2^24)
inline unsigned int lcg(unsigned int &prev)
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}


// Generate random float in [0, 1)
inline float rnd(unsigned int &prev)
{
    return ((float) lcg(prev) / (float) 0x01000000);
}


inline float RI_vdC( unsigned bits, unsigned r)
{
    bits = ( bits << 16)
         | ( bits >> 16);
    bits = ((bits & 0x00ff00ff) << 8)
         | ((bits & 0xff00ff00) >> 8);
    bits = ((bits & 0x0f0f0f0f) << 4)
         | ((bits & 0xf0f0f0f0) >> 4);
    bits = ((bits & 0x33333333) << 2)
         | ((bits & 0xcccccccc) >> 2);
    bits = ((bits & 0x55555555) << 1)
         | ((bits & 0xaaaaaaaa) >> 1);
    bits ^= r;
    return static_cast<float>( bits ) / ( static_cast<float>( UINT_MAX ) + 1.0f );
}


inline float RI_S(unsigned i, unsigned r )
{
    for(unsigned v = 1<<31; i; i >>= 1, v ^= v>>1)
        if(i & 1) r ^= v;
    return static_cast<float>( r ) / ( static_cast<float>( UINT_MAX ) + 1.0f );
}


inline float RI_LP(unsigned i, unsigned r)
{
    for(unsigned v = 1<<31; i; i >>= 1, v |= v>>1)
        if(i & 1) r ^= v;
    return static_cast<float>( r ) / ( static_cast<float>( UINT_MAX ) + 1.0f );
}


// TODO specialize this for base  3, 5 
inline float RI_hlt( unsigned i, unsigned prime )
{
    float h = 0.0f;
    float f = 1.0f/static_cast<float>( prime );
    float fct = 1.0f;
    while( i > 0 )
    {
        fct *= f; 
        h += (i%prime)*fct;
        i /= prime;
    }
    return h;
}


// Sobol sequence
inline Vector2 sobol( unsigned i, unsigned r )
{
    return Vector2( RI_vdC(i, r ), RI_S(i, r ) );
}

// Halton base 3,5
inline Vector2 halton( unsigned i )
{
    return Vector2( RI_hlt(i, 3), RI_hlt(i, 5) ); 
}


// Best but requires knowledge of total number of samples/pixel a priori (spp
// should be power of 2 as well )
inline Vector2 hammersley( unsigned i, unsigned spp, unsigned r )
{
    return Vector2( (float)i / (float)spp, RI_LP(i, r ) ); 
}


// Multiply with carry
inline unsigned int mwc()
{
    static unsigned long long r[4];
    static unsigned long long carry;
    static bool init = false;
    if( !init ) {
        init = true;
        unsigned int seed = 7654321u, seed0, seed1, seed2, seed3;
        r[0] = seed0 = lcg(seed);
        r[1] = seed1 = lcg(seed0);
        r[2] = seed2 = lcg(seed1);
        r[3] = seed3 = lcg(seed2);
        carry = lcg(seed3);
    }

    unsigned long long sum = 2111111111ull * r[3] +
        1492ull       * r[2] +
        1776ull       * r[1] +
        5115ull       * r[0] +
        1ull          * carry;
    r[3]   = r[2];
    r[2]   = r[1];
    r[1]   = r[0];
    r[0]   = static_cast<unsigned int>(sum);        // lower half
    carry  = static_cast<unsigned int>(sum >> 32);  // upper half
    return static_cast<unsigned int>(r[0]);
}


inline unsigned int random1u()
{
#if 0
    return rand();
#else
    return mwc();
#endif
}


inline Index2 random2u()
{
    return Index2( random1u(), random1u() );
}


inline unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}

}

#endif // LEGION_COMMON_MATH_HPP_
