
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


#include <gui/RenderThread.hpp>
#include <gui/LegionDisplay.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Objects/Renderer/IRenderer.hpp>

#include <QImage>

using namespace lr;

RenderThread::RenderThread( legion::Context* ctx )
    : QThread(),
      m_ctx( ctx )
{
    
    if( !m_ctx->getRenderer()->getDisplay() )
    {
        m_display = new LegionDisplay( m_ctx, this, "lrgui.exr" );
        m_ctx->getRenderer()->setDisplay( m_display );
    }

    m_ctx->getRenderer()->getDisplay()->beginScene( "lrgui");
    legion::Index2 res = m_ctx->getRenderer()->getResolution();
    m_image = new QImage( res.x(), res.y(), QImage::Format_RGB32 );
}


RenderThread::~RenderThread()
{
}


void RenderThread::emitImageUpdated(
        const unsigned char* image,
        int percent_done
        )
{
    emit imageUpdated( image, percent_done );
}


void RenderThread::emitImageFinished()
{
    emit imageFinished();
}


void RenderThread::run()
{
    m_ctx->render();
}
