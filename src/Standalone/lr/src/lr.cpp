
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
#include <Util.hpp>
#include <XMLToLegion.hpp>
#include <gui/GUI.hpp>

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
              << "Options:\n"
              << "\t--nogui    Turn off gui mode\n"
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
  std::cout << "HELLO" << std::endl;
    if( argc < 2 )
        printUsageAndExit( argv[0] );

    bool use_gui = true;
    for( int i = 1; i < argc-1; ++i )
    {
        std::string arg( argv[i] );
        if( arg == "--nogui" )
        {
            use_gui = false;
        }
        else
        {
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        lr::XMLToLegion translate( lr::parseScene(argv[argc-1]), use_gui );
        if( !use_gui )
        {
          std::cout << "*******************************BBBBBBBBBBBBBBBB" << std::endl;
            translate.getContext()->render();
        }
        else
        {
          std::cout << "*******************************AAAAAAAAAAAAAAA" << std::endl;
            lr::GUI( translate.getContext(), argc, argv );
        }

    }
    catch( std::exception& e )
    {
        std::cout << "Parsing failed: " << e.what() << std::endl;
        return 0;
    }
}


