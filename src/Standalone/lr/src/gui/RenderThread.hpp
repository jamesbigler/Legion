
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


#ifndef LR_GUI_RENDER_THREAD_HPP_
#define LR_GUI_RENDER_THREAD_HPP_

#include <QThread>

class QImage;

namespace legion
{
    class Context;
}

namespace lr
{

class LegionDisplay;

class RenderThread: public QThread
{
    Q_OBJECT

public:

    RenderThread( legion::Context* ctx );
    ~RenderThread();

    void emitImageUpdated ( QImage* image, int percent_done );
    void emitImageFinished( QImage* image );

signals:
    void imageUpdated ( QImage*, int percent_done );
    void imageFinished( QImage* );

protected:
    void run();

private:
    legion::Context*    m_ctx;
    lr::LegionDisplay*  m_display;
};

}

#endif // LR_GUI_RENDER_THREAD_HPP_
