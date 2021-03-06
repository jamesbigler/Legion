
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


#include <gui/DisplayWidget.hpp>
#include <gui/ImageWidget.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Util/Timer.hpp>
#include <iostream>


using namespace lr;


DisplayWidget::DisplayWidget()
    : QWidget(),
      m_image_widget( new ImageWidget( this ) ),
      m_image( 0 ),
      m_width( 512 ),
      m_height( 512 )
{
    m_image_widget->resize( m_width, m_height );
}


DisplayWidget::~DisplayWidget()
{
}

    
void DisplayWidget::setResolution( unsigned width, unsigned height )
{
    m_width  = width;
    m_height = height;
    m_image  = new QImage( m_width, m_height, QImage::Format_RGB32 );
    parentWidget()->resize( m_width, m_height );
    m_image_widget->resize( m_width, m_height );
}


void DisplayWidget::displayImage( const unsigned char* image )
{
    struct ARGB { unsigned char a,r,g,b; };

    legion::Timer timer;
    timer.start();
    const ARGB* pixels = reinterpret_cast<const ARGB*>( image );

    for( int i = 0; i < m_width; ++i )
    {
        for( int j = 0; j < m_height; ++j )
        {
            unsigned idx = (j*m_width+i);
            const ARGB c = pixels[ idx ];
            QRgb color = qRgb( c.r, c.g, c.b );

            m_image->setPixel( i, j, color );

        }
    }
    m_image_widget->setPixmap( QPixmap::fromImage( *m_image ) );
}
