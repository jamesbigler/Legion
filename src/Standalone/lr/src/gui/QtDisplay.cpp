
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


#include <gui/QtDisplay.hpp>

/////////////////////////////////////////////////////////////////////////
#include <QApplication>
#include <QWidget>
/////////////////////////////////////////////////////////////////////////


using namespace lr;


QtDisplay::QtDisplay( legion::Context* context )
    : legion::IDisplay( context ),
      m_update_count( 0u ),
      m_cur_update( 0u ),
      m_qt_argc( 1 )
{
    m_qt_argv    = new char*[1];
    m_qt_argv[0] = new char [3];
    m_qt_argv[0][0] = 'l';
    m_qt_argv[0][1] = 'r';
    m_qt_argv[0][2] = '\0';
}


QtDisplay::~QtDisplay()
{
    delete [] m_qt_argv[0];
    delete [] m_qt_argv;
}


void QtDisplay::beginScene( const std::string& scene_name )
{
    m_scene_name = scene_name;
}


void QtDisplay::setUpdateCount( unsigned update_count )
{
    m_update_count = update_count;
}


void QtDisplay::beginFrame()
{
    QApplication app( m_qt_argc, m_qt_argv );

    QWidget window;

    window.resize(250, 150);
    window.setWindowTitle("Simple example");
    window.show();

    app.exec();
}

void QtDisplay::updateFrame( const legion::Index2&, const float* )
{
}

void QtDisplay::completeFrame( const legion::Index2& ,
                               const float* )
{
}

