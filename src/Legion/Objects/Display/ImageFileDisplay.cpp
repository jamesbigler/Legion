
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


#include <Legion/Objects/Display/ImageFileDisplay.hpp>
#include <Legion/Common/Util/Image.hpp>
#include <iostream>
#include <iomanip>

using namespace legion;

ImageFileDisplay::ImageFileDisplay( Context* context, const char* filename )
    : IDisplay( context ),
      m_update_count( 0u ),
      m_cur_update( 0u ),
      m_filename( filename )
{
}


void ImageFileDisplay::beginScene( const std::string& scene_name )
{
    m_scene_name = scene_name;
    std::cout << "\n"
              << " Legion rendering library                     v0.1\n"
              << " -------------------------------------------------\n"
              << std::endl;

    std::cout << "     Rendering Scene: '" << m_scene_name << "' ...\n"
              << std::endl;

    m_preprocess_timer.start();
}


void ImageFileDisplay::setUpdateCount( unsigned update_count )
{
    m_update_count = update_count;
}


void ImageFileDisplay::beginFrame()
{
    m_preprocess_timer.stop();
    m_render_timer.start();

    std::cout << " Scene Creation Time:" 
                  << std::setiosflags(std::ios::right) 
                  << std::setw( s_field_width - 1 ) 
                  << std::setiosflags( std::ios::fixed)
                  << std::setprecision( 2 )
                  << m_preprocess_timer.getTimeElapsed() << "s\n";

    if( m_update_count > 0 )
    {
        std::cout << "     Render Progress:" 
                  << std::setiosflags(std::ios::right) 
                  << std::setw( s_field_width ) 
                  << "0%";
        std::cout.flush();
    }
    else
    {
        std::cout << "        Update Count:" 
                  << std::setw( s_field_width ) 
                  << std::setiosflags(std::ios::right) 
                  << "0";
        std::cout.flush();
    }
    
}

void ImageFileDisplay::updateFrame( const Index2&, const float* )
{
    ++m_cur_update;
    if( m_update_count > 0 )
    {
        const unsigned percent_done = 
            static_cast<float>( m_cur_update )   /
            static_cast<float>( m_update_count ) *
            100.0f;
        std::cout << "\b\b\b\b" 
                  << std::setw( 3 ) 
                  << std::setiosflags(std::ios::right ) 
                  << percent_done << "%";
        std::cout.flush();
    }
    else
    {
        std::cout << "\b\b\b\b" 
                  << std::setw( 4 ) 
                  << std::setiosflags(std::ios::right) 
                  << m_cur_update;
        std::cout.flush();
    }
}

void ImageFileDisplay::completeFrame( const Index2& resolution,
                                      const float* pixels )
{
    m_render_timer.stop();
    m_postprocess_timer.start();

    if( m_update_count > 0 )
    {
        std::cout << "\b\b\b\b100%" << std::endl; 
    }
    else
    {
        std::cout << std::endl; 
    }

    // TODO: Wrap this common time IO stuff into helper
    std::cout << "         Render Time:" 
                  << std::setiosflags(std::ios::right) 
                  << std::setw( s_field_width - 1 ) 
                  << std::setiosflags( std::ios::fixed)
                  << std::setprecision( 2 )
                  << m_render_timer.getTimeElapsed() << "s"
                  << std::endl;

    writeOpenEXR( m_filename, 
                  resolution.x(), 
                  resolution.y(), 
                  4,
                  pixels );

    m_postprocess_timer.stop();
    std::cout << "     Cleanup/IO Time:" 
                  << std::setiosflags(std::ios::right) 
                  << std::setw( s_field_width - 1 ) 
                  << std::setiosflags( std::ios::fixed)
                  << std::setprecision( 2 )
                  << m_postprocess_timer.getTimeElapsed() << "s"
                  << std::endl;
    std::cout << "          Total Time:" 
                  << std::setiosflags(std::ios::right) 
                  << std::setw( s_field_width - 1 ) 
                  << std::setiosflags( std::ios::fixed)
                  << std::setprecision( 2 )
                  << m_preprocess_timer.getTimeElapsed()   +
                      m_render_timer.getTimeElapsed()      +
                      m_postprocess_timer.getTimeElapsed()
                  << "s\n"
                  << std::endl;

    std::cout << "        Output Image:" 
                  << std::setw( s_field_width ) 
                  << m_filename
                  << std::endl << std::endl;;
}

