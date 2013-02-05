 

// Copyright (C) 2011 R. Keith Morley
//
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
// (MIT/X11 License)

#include <Legion/Common/Util/Image.hpp>
#include <Legion/Core/Exception.hpp>
#include <ImfArray.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>



using namespace legion;


bool legion::writeOpenEXR( const std::string& filename,
                   unsigned int width,
                   unsigned int height,
                   unsigned int num_channels,
                   const float* pixels )
{
    if( num_channels < 3 || num_channels > 4 )
    {
        throw Exception( std::string( __PRETTY_FUNCTION__ ) + ": Only 3-channel"
                         " and 4-channel images supported - RGB(A)" );
    }

    Imf::Header header( width, height );
    header.channels().insert ("R", Imf::Channel( Imf::FLOAT ) );
    header.channels().insert ("G", Imf::Channel( Imf::FLOAT ) );
    header.channels().insert ("B", Imf::Channel( Imf::FLOAT ) );
    if( num_channels == 4 )
      header.channels().insert ("A", Imf::Channel( Imf::FLOAT ) );


    Imf::OutputFile file( filename.c_str(), header );
    Imf::FrameBuffer frame_buffer; 
    const unsigned int element_stride = sizeof(float)*num_channels;

    frame_buffer.insert( "R", Imf::Slice( Imf::FLOAT,                // type
                                          (char*)&pixels[0],         // base
                                          element_stride,            // xStride
                                          element_stride*width ) );  // yStride

    frame_buffer.insert( "G", Imf::Slice( Imf::FLOAT,                // type
                                          (char*)&pixels[1],         // base
                                          element_stride,            // xStride
                                          element_stride*width ) );  // yStride

    frame_buffer.insert( "B", Imf::Slice( Imf::FLOAT,                // type
                                          (char*)&pixels[2],         // base
                                          element_stride,            // xStride
                                          element_stride*width ) );  // yStride
    if( num_channels == 4 )
    {
      frame_buffer.insert( "A", Imf::Slice( Imf::FLOAT,               // type
                                            (char*)&pixels[3],        // base
                                            element_stride,           // xStride
                                            element_stride*width ) ); // yStride
    }

    file.setFrameBuffer( frame_buffer );
    file.writePixels( height );

    return true;
}


bool legion::readOpenEXR(  const std::string& filename,
                           optix::Buffer buffer )
{

    Imf::RgbaInputFile file( filename.c_str() );

    Imath::Box2i dw  = file.dataWindow();
    unsigned width  = dw.max.x - dw.min.x + 1;
    unsigned height = dw.max.y - dw.min.y + 1;
    Imf::Array2D<Imf::Rgba> pixels;
    pixels.resizeErase( height, width );

    file.setFrameBuffer( &pixels[0][0] - dw.min.x - dw.min.y*width, 1, width );
    file.readPixels( dw.min.y, dw.max.y );
    
    buffer->setFormat( RT_FORMAT_FLOAT4 );
    buffer->setSize( width, height );

    optix::float4* data = reinterpret_cast<optix::float4*>( buffer->map() );
    for( unsigned i = 0; i < width; ++i )
    {
        for( unsigned j = 0; j < height; ++j )
        {
            unsigned idx = ((height-1-j)*width + i);

            data[idx].x = pixels[j][i].r;
            data[idx].y = pixels[j][i].g;
            data[idx].z = pixels[j][i].b;
            data[idx].w = pixels[j][i].a;
        }
    }

    buffer->unmap();
    return true;
}
