
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

#ifndef LR_GUI_DISPLAY_WIDGET_HPP_
#define LR_GUI_DISPLAY_WIDGET_HPP_

#include <Legion/Objects/Display/IDisplay.hpp>
#include <string>
#include <QWidget>


class QImage;

namespace lr
{

class ImageWidget;

class DisplayWidget : public QWidget
{
    Q_OBJECT

public:
    DisplayWidget();
    ~DisplayWidget();

    void setResolution( unsigned width, unsigned height );

private slots:
    void displayImage( const float* );

private:
    ImageWidget*      m_image_widget;
    QImage*           m_image;

    unsigned          m_width;
    unsigned          m_height;
};

} // end namespace lr

#endif // LR_GUI_DISPLAY_WIDGET_HPP_
