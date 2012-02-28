 

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
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfRgbaFile.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>



using namespace legion;


bool legion::writeOpenEXR( const std::string& filename,
                   unsigned int width,
                   unsigned int height,
                   const float* pixels )
{
    Imf::Header header( width, height );
    header.channels().insert ("R", Imf::Channel( Imf::FLOAT ) );
    header.channels().insert ("G", Imf::Channel( Imf::FLOAT ) );
    header.channels().insert ("B", Imf::Channel( Imf::FLOAT ) );

    Imf::OutputFile file( filename.c_str(), header );
    Imf::FrameBuffer frame_buffer; 

    frame_buffer.insert( "R", Imf::Slice( Imf::FLOAT,                // type
                                          (char*)&pixels[0],         // base
                                          sizeof(float)*3,           // xStride
                                          sizeof(float)*3*width ) ); // yStride

    frame_buffer.insert( "G", Imf::Slice( Imf::FLOAT,                // type
                                          (char*)&pixels[1],         // base
                                          sizeof(float)*3,           // xStride
                                          sizeof(float)*3*width ) ); // yStride

    frame_buffer.insert( "B", Imf::Slice( Imf::FLOAT,                // type
                                          (char*)&pixels[2],         // base
                                          sizeof(float)*3,           // xStride
                                          sizeof(float)*3*width ) ); // yStride

    file.setFrameBuffer( frame_buffer );
    file.writePixels( height );

    return true;
}
