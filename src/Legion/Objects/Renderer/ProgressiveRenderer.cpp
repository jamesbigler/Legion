
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


void ProgressiveRenderer::render()
{
    // TODO: iterate over number of samples
    // TODO: update display
    Timer timer;

    for( int i = 0; i < 10; ++i )
    {
      AutoPrintTimer apt( PrintTimeElapsed( "\tOptiX launch" ) );
      m_output_buffer->setSize( getResolution().x(), getResolution().y() );
      launchOptiX( getResolution() );
    }

    AutoPrintTimer apt( PrintTimeElapsed( "\tDisplay" ) );
    m_display->completeFrame( getResolution(), reinterpret_cast<float*>( m_output_buffer->map() ) );
    m_output_buffer->unmap();

}
    

void ProgressiveRenderer::setVariables( VariableContainer& container ) const
{
    container.setBuffer( "legion_output_buffer", m_output_buffer );
}

