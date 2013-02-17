
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


#ifndef LR_GUI_WINDOW_HPP_
#define LR_GUI_WINDOW_HPP_

#include <QMainWindow>
#include <string>



class QImage;
class QProgressBar;
class QPushButton;
class QStatusBar;
class QToolBar;

namespace legion
{
    class Context;
}

namespace lr
{

class DisplayWidget;
class RenderThread; 

class Window : public QMainWindow
{
    Q_OBJECT

public:
    Window( const char* filename );
    ~Window();

private slots:
    void loadScene();
    void render();
    void imageUpdated ( const unsigned char*, int );
    void imageFinished();

signals:
    void appStarting();
    void sceneLoaded();
    void imageUpdated( const unsigned char* );

private:
    std::string       m_filename;
    legion::Context*  m_ctx;

    DisplayWidget*    m_display_widget;
    RenderThread*     m_render_thread;
    
    QPushButton*      m_render_button;
    QPushButton*      m_stop_button;
    QProgressBar*     m_progress_bar;
    QToolBar*         m_tool_bar;
    QStatusBar*       m_status_bar;
};

}

#endif // LR_GUI_WINDOW_HPP_
