
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
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


#include <iostream>
#include <QApplication>
#include <gui/Window.hpp>

//#include <QPushButton>

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 );

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cout << "\nUsage  : " << argv0 << " [options] <scene_file.xml>\n\n"
              << std::endl;
    exit(0);
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

int main( int argc , char** argv )
{
    if( argc < 2 )
        printUsageAndExit( argv[0] );

    /*
    QApplication app(argc, argv);
    QPushButton helloButton("Hello World");
    helloButton.resize(80, 20);
    helloButton.show();
    app.exec();
    */

    QApplication app( argc, argv );

    lr::Window* window = new lr::Window( argv[ argc-1 ] );
    window->show();

    app.exec();

}


