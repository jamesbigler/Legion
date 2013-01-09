

#include <Legion/Common/Util/Image.hpp>
#include <cstdlib>

int main( int, char** )
{
    const unsigned num_channels = 3;
    const unsigned width  = 256;
    const unsigned height = width;
    const unsigned num_pixels = width*height;

    float* pixels = new float[ num_pixels*3 ];
    for( unsigned i = 0; i < num_pixels*num_channels; i+=num_channels )
    {
        pixels[i+0] = static_cast<float>( i ) /
                      static_cast<float>( num_pixels*num_channels ); 
        pixels[i+1] = 0.0f; 
        pixels[i+2] = static_cast<float>( num_pixels*num_channels-i ) /
                      static_cast<float>( num_pixels*num_channels )*100.0f; 
    }

    legion::writeOpenEXR( "test.exr", width, height, num_channels, pixels );

}
