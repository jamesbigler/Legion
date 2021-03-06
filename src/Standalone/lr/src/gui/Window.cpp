
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


#include <Util.hpp>
#include <XMLToLegion.hpp>
#include <gui/DisplayWidget.hpp>
#include <gui/ImageWidget.hpp>
#include <gui/RenderThread.hpp>
#include <gui/Window.hpp>

#include <iostream>

#include <QPushButton>
#include <QStatusBar>
#include <QProgressBar>
#include <QToolBar>
#include <QTimer>

using namespace lr;

Window::Window( const char* filename )
    : QMainWindow(),
      m_filename( filename ),
      m_ctx( new legion::Context )
{
    const std::string scene_file = m_filename;
    const size_t pos = scene_file.find_last_of( '/' );
    const std::string scene_dir = pos == std::string::npos ?
        "."                      :
        scene_file.substr( 0, pos );
    m_ctx->addAssetPath( scene_dir );


    m_display_widget = new DisplayWidget();
    setCentralWidget( m_display_widget );
    setWindowTitle( tr( "lr v0.1") );
    resize( 800, 600 );
    move( 0, 0 );

    /*
    QPalette pal = this->palette();
    pal.setColor( this->backgroundRole(), Qt::blue);
    this->setPalette(pal);
    */
    //setBackgroundRole( QPalette::Dark );
      
    m_status_bar = new QStatusBar( this );
    setStatusBar( m_status_bar );

    m_render_button = new QPushButton( tr("Render") );
    m_stop_button   = new QPushButton( tr("Stop"  ) );
    m_progress_bar  = new QProgressBar( this );
    m_progress_bar->setRange( 0, 100 );

    m_tool_bar = new QToolBar( tr("Toolbar"), this );
    m_tool_bar->addWidget( m_progress_bar  );
    m_tool_bar->addWidget( m_render_button );
    m_tool_bar->addWidget( m_stop_button   );
    addToolBar( Qt::TopToolBarArea, m_tool_bar );

    QTimer::singleShot( 0, this, SIGNAL( appStarting() ) );

    connect( this, SIGNAL( appStarting() ), this, SLOT( loadScene() ) );
    connect( this, SIGNAL( sceneLoaded() ), this, SLOT( render() ) );

    statusBar()->showMessage( tr("Loading scene...") );
}


Window::~Window()
{
}


void Window::loadScene()
{
    statusBar()->showMessage( tr("Loading scene...") );

    // Read in and translate XML text
    char* text;
    if( ! lr::readFile( m_filename.c_str(), &text ) )
        throw std::runtime_error( "Failed to read xml file." );
    XMLToLegion translate( text, m_ctx, false );
    delete [] text;

    emit sceneLoaded();
}


void Window::imageUpdated( const unsigned char* image, int percent_done)
{
    m_progress_bar->setValue( percent_done );
    emit imageUpdated( image );
}

void Window::imageFinished()
{
    m_progress_bar->setValue( 100 );

    //
    // clean up render thread 
    //
    disconnect(
            m_render_thread, SIGNAL(imageUpdated(const unsigned char*, int)),
            this,            SLOT  (imageUpdated(const unsigned char*, int))
            );
    
    disconnect( m_render_thread, SIGNAL(imageFinished() ),
                this,            SLOT  (imageFinished() ) );

    m_render_thread->wait();
    m_render_thread->quit();
    delete m_render_thread;
    m_render_thread = 0;
}


void Window::render()
{
    statusBar()->showMessage( tr("Rendering...") );
    m_progress_bar->setValue( 0 );
    legion::IRenderer* renderer = m_ctx->getRenderer();
    legion::Index2 resolution = renderer->getResolution();
    m_display_widget->setResolution( resolution.x(), resolution.y() );

    m_render_thread = new RenderThread( m_ctx );

    connect( m_render_thread, SIGNAL(imageUpdated(const unsigned char*, int)),
             this,            SLOT  (imageUpdated(const unsigned char*, int)) );
    
    connect( m_render_thread, SIGNAL(imageFinished()),
             this,            SLOT  (imageFinished()) );

    connect( this,             SIGNAL( imageUpdated(const unsigned char* ) ),
             m_display_widget, SLOT  ( displayImage(const unsigned char* ) ) );

    m_render_thread->start();
}

