
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

#include <cstdlib>

using namespace legion;
    

IRenderer* ProgressiveRenderer::create( Context* context, const Parameters& )
{
    return new ProgressiveRenderer( context );
}


ProgressiveRenderer::ProgressiveRenderer( Context* context ) 
   : IRenderer( context )
{
    m_float_output_buffer = createOptiXBuffer( RT_BUFFER_OUTPUT,
                                               RT_FORMAT_FLOAT4 );
    m_byte_output_buffer  = createOptiXBuffer( RT_BUFFER_OUTPUT,
                                               RT_FORMAT_UNSIGNED_BYTE4);
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


void ProgressiveRenderer::render( VariableContainer& container )
{
    LLOG_INFO << "\tProgressiveRenderer : rendering ...";
    m_float_output_buffer->setSize( getResolution().x(), getResolution().y() );
    m_byte_output_buffer->setSize ( getResolution().x(), getResolution().y() );

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
    m_display->beginFrame( getResolution() );

    for( unsigned i = 0; i < getSamplesPerPixel(); ++i )
    {
        LoopTimer timer( progressive_updates );

        container.setUnsigned( "sample_index", i         );
        container.setFloat   ( "light_seed"  , drand48() );
        launchOptiX( getResolution() );

        const IDisplay::FrameType frame_type = m_display->getUpdateFrameType();
        float* fpix = 0;
        unsigned char* bpix = 0;
        if( frame_type & IDisplay::FLOAT )
            fpix = reinterpret_cast<float*>( m_float_output_buffer->map() );
        if( frame_type & IDisplay::BYTE )
            bpix =
                reinterpret_cast<unsigned char*>( m_byte_output_buffer->map() );

        bool keep_going = m_display->updateFrame( fpix, bpix );
        
        if( frame_type & IDisplay::FLOAT )
            m_float_output_buffer->unmap();
        if( frame_type & IDisplay::BYTE )
            m_byte_output_buffer->unmap();

        if( !keep_going )
            break;
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
    const IDisplay::FrameType frame_type = m_display->getCompleteFrameType();
    float* fpix = 0;
    unsigned char* bpix = 0;
    if( frame_type & IDisplay::FLOAT )
        fpix = reinterpret_cast<float*>( m_float_output_buffer->map() );
    if( frame_type & IDisplay::BYTE )
        bpix = reinterpret_cast<unsigned char*>( m_byte_output_buffer->map() );

    m_display->completeFrame( fpix, bpix ); 
    
    if( frame_type & IDisplay::FLOAT )
        m_float_output_buffer->unmap();
    if( frame_type & IDisplay::BYTE )
        m_byte_output_buffer->unmap();
}
    

void ProgressiveRenderer::setVariables( VariableContainer& container ) const
{
    const unsigned do_byte_updates  = m_display->getUpdateFrameType() &
                                      IDisplay::BYTE;
    const unsigned do_byte_complete = m_display->getCompleteFrameType() &
                                      IDisplay::BYTE;
    container.setBuffer  ( "float_output_buffer", m_float_output_buffer );
    container.setBuffer  ( "byte_output_buffer",  m_byte_output_buffer  );
    container.setUnsigned( "samples_per_pixel",   getSamplesPerPixel()  );
    container.setUnsigned( "do_byte_updates",     do_byte_updates       );
    container.setUnsigned( "do_byte_complete",    do_byte_complete      );
}

