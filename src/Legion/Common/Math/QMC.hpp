

/*
const unsigned LENS_SEED   = 0x0000c365u;   // 50021
const unsigned PIXEL_SEED  = 0x00118c4bu;   // 1150027
const unsigned SHADOW_SEED = 0x009ba3c7u;   // 10200007
const unsigned BSDF_SEED   = 0x3bbb98bdu;   // 1002150077
*/

const unsigned BSDF_SEED   = 0x9e3779b9u;
const unsigned PIXEL_SEED  = 0xc8013ea4u;
const unsigned SHADOW_SEED = 0xad90777du;
const unsigned LENS_SEED   = 0x7e95761eu;


template<unsigned int N>
__host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
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
__host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

__host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
  prev = (prev*8121 + 28411)  % 134456;
  return prev;
}

// Generate random float in [0, 1)
__host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}


__device__ __inline__ 
float RI_vdC(uint bits, uint r)
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


__device__ __inline__ 
float RI_S(uint i, uint r )
{
    for(uint v = 1<<31; i; i >>= 1, v ^= v>>1)
        if(i & 1) r ^= v;
    return static_cast<float>( r ) / ( static_cast<float>( UINT_MAX ) + 1.0f );
}


__device__ __inline__ 
float RI_LP(uint i, uint r)
{
    for(uint v = 1<<31; i; i >>= 1, v |= v>>1)
        if(i & 1) r ^= v;
    return static_cast<float>( r ) / ( static_cast<float>( UINT_MAX ) + 1.0f );
}


// TODO specialize this for base  3, 5 
__device__ __inline__ 
float RI_hlt( uint i, uint prime )
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
__device__ __inline__ 
float2 lds_rnd( uint i, uint r )
{
     return make_float2( RI_vdC(i, r ), RI_S(i, r ) );
}

// Halton base 3,5
__device__ __inline__ 
float2 lds_rnd2( uint i )
{
    return make_float2( RI_hlt(i, 3), RI_hlt(i, 5) ); 
}

// Best but requires knowledge of total number of samples/pixel a priori (spp should be power of 2 as well )
__device__ __inline__ 
float2 lds_rnd2( uint i, uint spp, uint r = 0 )
{
    return make_float2( (float)i / (float)spp, RI_LP(i, r ) ); 
}

// Multiply with carry
__host__ __inline__ unsigned int mwc()
{
  static unsigned long long r[4];
  static unsigned long long carry;
  static bool init = false;
  if( !init ) {
    init = true;
    unsigned int seed = 7654321u, seed0, seed1, seed2, seed3;
    r[0] = seed0 = lcg2(seed);
    r[1] = seed1 = lcg2(seed0);
    r[2] = seed2 = lcg2(seed1);
    r[3] = seed3 = lcg2(seed2);
    carry = lcg2(seed3);
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


__host__ __inline__ unsigned int random1u()
{
#if 0
  return rand();
#else
  return mwc();
#endif
}

__host__ __inline__ optix::uint2 random2u()
{
  return optix::make_uint2(random1u(), random1u());
}

__host__ __inline__ void fillRandBuffer( unsigned int *seeds, unsigned int N )
{
  for( unsigned int i=0; i<N; ++i ) 
    seeds[i] = mwc();
}

__host__ __device__ __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}
