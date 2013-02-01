
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


#include <Legion/Objects/Renderer/ProgressiveRenderer.hpp>
#include <Legion/Objects/Display/IDisplay.hpp>
#include <Legion/Core/VariableContainer.hpp>
#include <Legion/Common/Util/AutoTimerHelpers.hpp>
#include <Legion/Common/Util/Logger.hpp>

using namespace legion;

ProgressiveRenderer::ProgressiveRenderer( Context* context ) 
   : IRenderer( context )
{
    m_output_buffer = createOptiXBuffer( RT_BUFFER_OUTPUT,
                                         RT_FORMAT_FLOAT4 );
}


ProgressiveRenderer::~ProgressiveRenderer()
{
}

    
const char* ProgressiveRenderer::name()const
{
    return "ProgressiveRenderer";
}


const char* ProgressiveRenderer::rayGenProgramName()const
{
    return "progressiveRendererRayGen";
}


optix::Buffer ProgressiveRenderer::getOutputBuffer()const
{
    return m_output_buffer;
}


void ProgressiveRenderer::render( VariableContainer& container )
{
    LLOG_INFO << "\tProgressiveRenderer : rendering ...";
    m_output_buffer->setSize( getResolution().x(), getResolution().y() );

    //
    // Force pre-compilation and accel structure builds
    //
    {
        container.setUnsigned( "sample_index", 0u   );
        container.setFloat   ( "light_seed"  , 0.0f );
        AutoPrintTimer apt( PrintTimeElapsed( "\tCompile/accel build " ) );
        launchOptiX( Index2( 0u, 0u ) );
    }

    //
    // Progressive loop
    //
    srand48( 12345678 );

    LoopTimerInfo progressive_updates( "\tProgressive loop    :" );
    m_display->setUpdateCount( getSamplesPerPixel() );
    m_display->beginFrame();
    for( unsigned i = 0; i < getSamplesPerPixel(); ++i )
    {
        LoopTimer timer( progressive_updates );

        container.setUnsigned( "sample_index", i         );
        container.setFloat   ( "light_seed"  , drand48() );
        launchOptiX( getResolution() );
        m_display->updateFrame( 
            getResolution(), 
            reinterpret_cast<float*>( m_output_buffer->map() ) );
        m_output_buffer->unmap();
    }

    //
    // Log statistics
    //
    progressive_updates.log();
    const float num_pixels = getResolution().x() * 
                             getResolution().y() * 
                             getSamplesPerPixel();
    LLOG_INFO << "\tTotal pixels/second : " 
              << static_cast<unsigned>( num_pixels / 
                                        progressive_updates.total_time );

    //
    // Display
    //
    AutoPrintTimer apt( PrintTimeElapsed( "\tDisplay" ) );
    m_display->completeFrame( 
        getResolution(), 
        reinterpret_cast<float*>( m_output_buffer->map() ) );
    m_output_buffer->unmap();
}
    

void ProgressiveRenderer::setVariables( const VariableContainer& container ) const
{
    container.setBuffer  ( "output_buffer",     m_output_buffer      );
    container.setBuffer  ( "output_buffer",     m_output_buffer      );
    container.setUnsigned( "samples_per_pixel", getSamplesPerPixel() );
}

