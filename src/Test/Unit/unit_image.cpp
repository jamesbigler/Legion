

#include <Legion/Common/Util/Image.hpp>
#include <cstdlib>

int main( int argc, char** argv )
{
    const unsigned width  = 256;
    const unsigned height = width;
    const unsigned num_pixels = width*height;

    float* pixels = new float[ num_pixels*3 ];
    for( unsigned i = 0; i < num_pixels*3; i+=3 )
    {
        pixels[i+0] = static_cast<float>( i ) / static_cast<float>( num_pixels*3 ); 
        //pixels[i+0] = drand48(); 
        //pixels[i+0] = 1.0f; 
        pixels[i+1] = 0.0f; 
        pixels[i+2] = static_cast<float>( num_pixels*3-i ) / static_cast<float>( num_pixels*3 ) *100.0f; 
    }

    legion::writeOpenEXR( "test.exr", width, height, pixels );

}
