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


#ifndef LEGION_COMMON_MATH_CUDA_PERLIN_NOISE_HPP_
#define LEGION_COMMON_MATH_CUDA_PERLIN_NOISE_HPP_

namespace legion
{


/// 
/// \class PerlinNoise
/// 
/// Perlin Noise is a pseudorandom scalar function defined on
/// n-dimensional space, described in the paper, "An Image
/// Synthesizer," by Ken Perlin from SIGGRAPH 1985.  Additional
/// information can also be found in the paper, "Improving Noise," also
/// by Perlin, from SIGGRAPH 2002.
/// 
/// In this implementation, noise is defined on R^3.  One- and
/// two-dimensional versions are available by projecting the
/// corresponding points "upward" into R^3.  All noise values lie in
/// the range [-1, 1].
/// 
/// Vector-valued noise is attained by taking the gradient of the
/// scalar noise function.  "Turbulent" noise is attained by adding
/// scaled versions of the scalar noise function at varying
/// frequencies.  Turbulent noise values lie in the range [-1, 1].
/// 
/// In this implementation, the noise, vector noise, and turbulent
/// noise functions are periodic with period 256 along all three
/// dimensions.  This version of noise is completely deterministic; no
/// random values are involved.
/// 
class PerlinNoise
{
public:
  LDEVICE static float  noise      (const float3& v );
  LDEVICE static float3 vectorNoise(const float3& v );
  LDEVICE static float  turbulence( const float3& v, unsigned int octaves);

private:
  const static float3       s_gradients[16];
  const static unsigned int s_permute[256];

  LDEVICE inline static float3 gradient( int i, int j, int k );
  LDEVICE inline static float blend ( float x );
  LDEVICE inline static float dBlend( float x );

};


/// Returns gradient vector for given three-dimensional integer lattice point.
LDEVICE inline float3 PerlinNoise::gradient(int i, int j, int k)
{
  return s_gradients[ 0xF & s_permute[ 0xFF & 
                      ( k + s_permute[ 0xFF & 
                      ( j + s_permute[ 0xFF & i ] ) ] ) ] ];
}


/// Blending function for interpolation of noise values.  See "Improving Noise"
/// by Perlin.
LDEVICE inline float PerlinNoise::blend( float x )
{
  // blending function is (10x^3 - 15x^4 + 6x^5)
  return (x*x*x*( 10.0f + x*( -15.0f + x*6.0f ) ) );
}


/// Derivative of the blending function above.
LDEVICE inline float PerlinNoise::dBlend( float x )
{
  return ( 30.0f*x*x*( 1.0f + x*( -2.0f + x ) ) );
}




/// Table of gradient vectors used in generating Perlin Noise.
const float3 legion::PerlinNoise::s_gradients[16] =
{
  make_float3(  1.0f,  1.0f,  0.0f ),
  make_float3( -1.0f,  1.0f,  0.0f ),
  make_float3(  1.0f, -1.0f,  0.0f ),
  make_float3( -1.0f, -1.0f,  0.0f ),
  make_float3(  1.0f,  0.0f,  1.0f ),
  make_float3( -1.0f,  0.0f,  1.0f ),
  make_float3(  1.0f,  0.0f, -1.0f ),
  make_float3( -1.0f,  0.0f, -1.0f ),
  make_float3(  0.0f,  1.0f,  1.0f ),
  make_float3(  0.0f, -1.0f,  1.0f ),
  make_float3(  0.0f,  1.0f, -1.0f ),
  make_float3(  0.0f, -1.0f, -1.0f ),
  make_float3(  1.0f,  1.0f,  0.0f ),
  make_float3( -1.0f,  1.0f,  0.0f ),
  make_float3(  0.0f, -1.0f,  1.0f ),
  make_float3(  0.0f, -1.0f, -1.0f )
};

/// Permutation used as a hash table for gradient vector lookups.
const unsigned PerlinNoise::s_permute[256] =
{
  241,  96,  11, 170, 166,  80, 123, 154,  28, 248, 212, 151,  12,  67, 201, 112,
   73,  70, 223, 144, 225,   0, 131,  50, 191, 202,  15, 138,  35, 161,  95, 252,
   21, 168, 226,  17,  33, 217, 200,  31, 146,   4, 132, 184, 203, 196,  10, 106,
  179, 176,  38, 245,  14, 232,  42, 194, 177,  23, 187,  65, 222,  34,  68,  69,
  186, 121, 246, 167,  84, 244, 188, 143,  88, 125,  79, 238,  54, 254, 224,   9,
   18,  51, 175, 158, 133,  91, 218, 172,  16, 165, 243, 141,  48, 105, 114,  99,
  205, 240,  49,  40, 163, 180, 204, 233, 242, 107,  62, 190, 249, 255,  56,  89,
  142,  93,  87, 156, 206,  81,  25,  86, 192,   8, 118,  36,   3, 139,  59,  55,
  104, 113, 214, 128, 174,  92, 149,  22, 134, 150, 122, 145,  39, 189,  44, 159,
  227, 148, 136,  27, 169, 164,  66,  32, 102, 109, 219, 239, 251,  74, 250, 185,
    7,  57,  29, 216,  61,  13, 181, 130,  64, 229, 237, 152, 116, 153,  19,  77,
   85, 115,  82,  43, 178, 129, 213,  83, 234, 111,  26,  20, 120, 135,  71,  37,
   41, 173, 230,  53, 195, 155, 160, 220, 236, 228,  30, 110, 171,  58, 235,  63,
  231, 103, 209,  78,  90, 208,   2, 127, 100, 183,  46, 197, 207,   6,  52,  60,
  193, 199, 101, 162,  94,  98, 215, 119,  72, 140,   1,  75, 124, 147,  24, 126,
  182, 247, 210,  97, 253, 157, 198, 108, 117, 137,   5,  76,  47, 211,  45, 221
};


/// This function returns a consistent, pseudorandom noise value for all points
//in 3-space.  All noise values lie in the range [-1, 1].
LDEVICE float PerlinNoise::noise( const float3& v )
{
  int3    f;
  float3  stu;
    
  f.x = static_cast<int>( floor( v.x ) );
  f.y = static_cast<int>( floor( v.y ) );
  f.z = static_cast<int>( floor( v.z ) );

  stu.x = v.x - f.x;
  stu.y = v.y - f.y;
  stu.z = v.z - f.z;
  
  using optix::dot;
  return legion::trilerp( blend( stu.x ), blend(stu.y ), blend( stu.z ),
		 dot( stu, gradient( f.x, f.y, f.z ) ), 
     dot( stu-make_float3(1.0f, 0.0f, 0.0f), gradient( f.x+1, f.y  , f.z  )),
     dot( stu-make_float3(0.0f, 1.0f, 0.0f), gradient( f.x  , f.y+1, f.z  )),
     dot( stu-make_float3(1.0f, 1.0f, 0.0f), gradient( f.x+1, f.y+1, f.z  )),
     dot( stu-make_float3(0.0f, 0.0f, 1.0f), gradient( f.x  , f.y  , f.z+1)),
     dot( stu-make_float3(1.0f, 0.0f, 1.0f), gradient( f.x+1, f.y  , f.z+1)),
     dot( stu-make_float3(0.0f, 1.0f, 1.0f), gradient( f.x  , f.y+1, f.z+1)),
     dot( stu-make_float3(1.0f, 1.0f, 1.0f), gradient( f.x+1, f.y+1, f.z+1)));
}


/**
 * This function returns the gradient of noise(v).
 **/
LDEVICE float3 PerlinNoise::vectorNoise(const float3& v) {
  int i, floors[3];
  float3 stu, tvNoise, grads[8];
  float blends[3], noiseVals[8];

  for (i = 0; i < 3; i++) {
    floors[i] = static_cast<int>(floor(v[i]));
    stu[i] = v[i] - floors[i];
    blends[i] = blend(stu[i]);
  }

  grads[0] = gradient(floors[0], floors[1], floors[2]);
  grads[1] = gradient(floors[0] + 1, floors[1], floors[2]);
  grads[2] = gradient(floors[0], floors[1] + 1, floors[2]);
  grads[3] = gradient(floors[0] + 1, floors[1] + 1, floors[2]);
  grads[4] = gradient(floors[0], floors[1], floors[2] + 1);
  grads[5] = gradient(floors[0] + 1, floors[1], floors[2] + 1);
  grads[6] = gradient(floors[0], floors[1] + 1, floors[2] + 1);
  grads[7] = gradient(floors[0] + 1, floors[1] + 1, floors[2] + 1);
  
  noiseVals[0] = dot(grads[0], stu);
  noiseVals[1] = dot(grads[1], stu - float3(1.0f, 0.0f, 0.0f));
  noiseVals[2] = dot(grads[2], stu - float3(0.0f, 1.0f, 0.0f));
  noiseVals[3] = dot(grads[3], stu - float3(1.0f, 1.0f, 0.0f));
  noiseVals[4] = dot(grads[4], stu - float3(0.0f, 0.0f, 1.0f));
  noiseVals[5] = dot(grads[5], stu - float3(1.0f, 0.0f, 1.0f));
  noiseVals[6] = dot(grads[6], stu - float3(0.0f, 1.0f, 1.0f));
  noiseVals[7] = dot(grads[7], stu - float3(1.0f, 1.0f, 1.0f));

  tvNoise[0] = dBlend(stu[0]) * GRBlerp(blends[1], blends[2],
					noiseVals[1] - noiseVals[0],
					noiseVals[3] - noiseVals[2],
					noiseVals[5] - noiseVals[4],
					noiseVals[7] - noiseVals[6]);
  
  tvNoise[1] = dBlend(stu[1]) * GRBlerp(blends[0], blends[2],
					noiseVals[2] - noiseVals[0],
					noiseVals[3] - noiseVals[1],
					noiseVals[6] - noiseVals[4],
					noiseVals[7] - noiseVals[5]);
  
  tvNoise[2] = dBlend(stu[2]) * GRBlerp(blends[0], blends[1],
					noiseVals[4] - noiseVals[0],
					noiseVals[5] - noiseVals[1],
					noiseVals[6] - noiseVals[2],
					noiseVals[7] - noiseVals[3]);
  
  return (tvNoise + GRTlerp(blend(stu[0]), blend(stu[1]), blend(stu[2]),
			    grads[0], grads[1], grads[2], grads[3],
			    grads[4], grads[5], grads[6], grads[7]));
}


/**
 * This function returns "turbulent" noise for each point in space.
 * The returned value is in the range [-1, 1].
 **/
LDEVICE float PerlinNoise::turbulence(const float3& v, unsigned int octaves) {
  unsigned int i;
  float t(0.0f), scale(1.0f);

  for (i = 0; i < octaves; i++) {
    t += noise(scale * v) / scale;
    scale *= 2.0f;
  }
  return (0.5 * t);
}

}

#endif // LEGION_COMMON_MATH_CUDA_PERLIN_NOISE_HPP_
