

#ifndef LEGION_COMMON_MATH_SOBOL_HPP_
#define LEGION_COMMON_MATH_SOBOL_HPP_
#include <climits>

namespace legion
{

class Sobol
{
public:
    
    static float gen( unsigned i, unsigned dim,  unsigned pixel_index = 0u )
    {
        const unsigned scramble = lcg( pixel_index ); 
        unsigned int m = ( dim & (MAX_DIMS-1u) ) << 5;
        unsigned int result = 0u;
        for( ; i; i >>= 1, ++m )
            result ^= MATRICES[m] * (i&1);
        result ^= scramble;
        return static_cast<float>( result ) / ( static_cast<float>( UINT_MAX ) + 1.0f );
    }

    static const unsigned MAX_DIMS = 128u;
    static const unsigned MATRICES[ MAX_DIMS*32u ];
private:
    static unsigned lcg( unsigned i ) { return i * 1664525u + 1013904223u; }
    Sobol();
};

}

#endif //  LEGION_COMMON_MATH_SOBOL_HPP_
