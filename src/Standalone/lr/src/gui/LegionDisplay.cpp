
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


#include <gui/LegionDisplay.hpp>
#include <Legion/Common/Util/Image.hpp>
#include <iostream>
#include <iomanip>

using namespace lr;



LegionDisplay::LegionDisplay( legion::Context* context,
                              RenderThread* render_thread,
                              const char* filename )
    : IDisplay( context ),
      m_render_thread( render_thread ),
      m_update_count( 0u ),
      m_cur_update( 0u ),
      m_filename( filename )
{
}



void LegionDisplay::setUpdateCount( unsigned update_count )
{
    m_update_count = update_count;
}


void LegionDisplay::beginScene( const std::string& scene_name )
{
    m_scene_name = scene_name;
    std::cout << "\n"
              << " Legion rendering library                     v0.1\n"
              << " -------------------------------------------------\n"
              << std::endl;

    std::cout << std::setw( s_field_width+2 ) 
              << "Rendering Scene: '" << m_scene_name << "' ...\n"
              << std::endl;

    m_preprocess_timer.start();
}


void LegionDisplay::beginFrame( const legion::Index2& resolution )
{
    m_resolution = resolution;

    printTime( "Scene Creation", m_preprocess_timer.stop() ); 
    m_render_timer.start();

    if( m_update_count > 0 )
    {
        std::cout << std::setw( s_field_width ) 
                  << "Render Progress:" 
                  << std::setiosflags(std::ios::right) 
                  << std::setw( s_result_width ) 
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


bool LegionDisplay::updateFrame( const float*, const Byte* bpix )
{
    ++m_cur_update;
    unsigned percent_done = 0u;

    if( m_update_count > 0 )
    {
        percent_done = 
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
    
    if( m_cur_update == 1u || m_display_timer.getTimeElapsed() > 0.5f )
    {
        m_render_thread->emitImageUpdated( bpix, percent_done );
        m_display_timer.reset();
        m_display_timer.start();
    }

    // TODO: allow user to stop render
    return true;
}


void LegionDisplay::completeFrame( const float* fpix, const Byte* )
{
    m_render_thread->emitImageFinished();

    if( m_update_count > 0 )
    {
        std::cout << "\b\b\b\b100%" << std::endl; 
    }
    else
    {
        std::cout << std::endl; 
    }

    printTime( "Render", m_render_timer.stop() );
    m_postprocess_timer.start();

    legion::writeOpenEXR(
            m_filename, m_resolution.x(), m_resolution.y(), 4, fpix
            );

    printTime( "Cleanup/IO", m_postprocess_timer.stop() );
    printTime( "Total Time",
            m_preprocess_timer.getTimeElapsed() +
            m_render_timer.getTimeElapsed()     +
            m_postprocess_timer.getTimeElapsed()
            );
    std::cout << "\n";

    std::cout << std::setw( s_field_width )
              << "Output Image:" 
              << std::setw( s_result_width ) 
              << m_filename
              << std::endl << std::endl;;
}

    
void LegionDisplay::printTime(
        const std::string& phase,
        double duration
        )
{

    std::cout << std::setw( s_field_width - 6 ) 
              << phase << " Time:"
              << std::setiosflags(std::ios::right) 
              << std::setw( s_result_width - 1 ) 
              << std::setiosflags( std::ios::fixed)
              << std::setprecision( 2 )
              << duration << "s"
              << std::endl;
}
