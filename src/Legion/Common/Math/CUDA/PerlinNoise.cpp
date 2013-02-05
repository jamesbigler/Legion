#include <math/PerlinNoise.h>
#include <utilities/GalileoFunctions.h>

/**
 * const Vector3 _gradients[16]
 *
 * Table of gradient vectors used in generating Perlin Noise.
 **/
const Vector3 PerlinNoise::_gradients[16] = {
  Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(-1.0, -1.0, 0.0),
  Vector3(1.0, 0.0, 1.0), Vector3(-1.0, 0.0, 1.0), Vector3(1.0, 0.0, -1.0), Vector3(-1.0, 0.0, -1.0),
  Vector3(0.0, 1.0, 1.0), Vector3(0.0, -1.0, 1.0), Vector3(0.0, 1.0, -1.0), Vector3(0.0, -1.0, -1.0),
  Vector3(1.0, 1.0, 0.0), Vector3(-1.0, 1.0, 0.0), Vector3(0.0, -1.0, 1.0), Vector3(0.0, -1.0, -1.0)
};

/**
 * const unsigned int _permute[256]
 *
 * Permutation used as a hash table for gradient vector lookups.
 **/
const unsigned int PerlinNoise::_permute[256] = {
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


/**
 * This function returns a consistent, pseudorandom noise value for
 * all points in 3-space.  All noise values lie in the range [-1, 1].
 **/
double PerlinNoise::noise(const Vector3& v) {
  int i, floors[3];
  Vector3 stu;
  
  for (i = 0; i < 3; i++) {
    floors[i] = static_cast<int>(floor(v[i]));
    stu[i] = v[i] - floors[i];
  }

  return GRTlerp(blend(stu[0]), blend(stu[1]), blend(stu[2]),
		 dot(stu, gradient(floors[0], floors[1], floors[2])), 
		 dot(stu - Vector3(1.0, 0.0, 0.0), gradient(floors[0] + 1, floors[1], floors[2])),
		 dot(stu - Vector3(0.0, 1.0, 0.0), gradient(floors[0], floors[1] + 1, floors[2])),
		 dot(stu - Vector3(1.0, 1.0, 0.0), gradient(floors[0] + 1, floors[1] + 1, floors[2])),
		 dot(stu - Vector3(0.0, 0.0, 1.0), gradient(floors[0], floors[1], floors[2] + 1)),
		 dot(stu - Vector3(1.0, 0.0, 1.0), gradient(floors[0] + 1, floors[1], floors[2] + 1)),
		 dot(stu - Vector3(0.0, 1.0, 1.0), gradient(floors[0], floors[1] + 1, floors[2] + 1)),
		 dot(stu - Vector3(1.0, 1.0, 1.0), gradient(floors[0] + 1, floors[1] + 1, floors[2] + 1)));
}


/**
 * This function returns the gradient of noise(v).
 **/
Vector3 PerlinNoise::vectorNoise(const Vector3& v) {
  int i, floors[3];
  Vector3 stu, tvNoise, grads[8];
  double blends[3], noiseVals[8];

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
  noiseVals[1] = dot(grads[1], stu - Vector3(1.0, 0.0, 0.0));
  noiseVals[2] = dot(grads[2], stu - Vector3(0.0, 1.0, 0.0));
  noiseVals[3] = dot(grads[3], stu - Vector3(1.0, 1.0, 0.0));
  noiseVals[4] = dot(grads[4], stu - Vector3(0.0, 0.0, 1.0));
  noiseVals[5] = dot(grads[5], stu - Vector3(1.0, 0.0, 1.0));
  noiseVals[6] = dot(grads[6], stu - Vector3(0.0, 1.0, 1.0));
  noiseVals[7] = dot(grads[7], stu - Vector3(1.0, 1.0, 1.0));

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
double PerlinNoise::turbulence(const Vector3& v, unsigned int octaves) {
  unsigned int i;
  double t(0.0), scale(1.0);

  for (i = 0; i < octaves; i++) {
    t += noise(scale * v) / scale;
    scale *= 2.0;
  }
  return (0.5 * t);
}
