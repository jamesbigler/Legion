
#ifndef LEGION_COMMON_MATH_RAND_HPP_
#define LEGION_COMMON_MATH_RAND_HPP_

#include <optixu/optixu_math_namespace.h>
#include <Legion/Common/Util/Preprocessor.hpp>


namespace legion
{
    template<unsigned int N>
    LDEVICE static inline unsigned tea( unsigned int val0, unsigned int val1 )
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

    // Generate random unsigned int in [0, 2^24) using simple LCG
    LDEVICE static inline unsigned int lcg( unsigned int &prev )
    {
        const unsigned int LCG_A = 1664525u;
        const unsigned int LCG_C = 1013904223u;
        prev = (LCG_A * prev + LCG_C);
        return prev & 0x00FFFFFF;
    }

    class LCGRand
    {
    public:
        LDEVICE LCGRand( unsigned seed ) 
            : m_seed( seed ) 
        {}

        LDEVICE float operator()()
        {
            return static_cast<float>( lcg( m_seed ) ) /
                   static_cast<float>(    0x01000000 );
        }

        LDEVICE unsigned getSeed()const 
        {
            return m_seed;
        }
    private:
        unsigned m_seed;
    };
}

#endif //LEGION_COMMON_MATH_RAND_HPP_
