
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
    if( argc < 2 )
        printUsageAndExit( argv[0] );

    bool use_gui = true;
    for( int i = 1; i < argc-1; ++i )
    {
        std::string arg( argv[i] );
        if( arg == "--nogui" )
            use_gui = false;
        else
            printUsageAndExit( argv[0] );
    }

    try
    {
        //
        // Read in xml text
        //
        char* text;
        if( ! lr::readFile( argv[ argc-1 ], &text ) )
            throw std::runtime_error( "Failed to read xml file." );

        //
        // Translate XML to legion data structure 
        //
        lr::XMLToLegion translate( text, use_gui );

        //
        // Run
        //
        if( !use_gui )
            translate.getContext()->render();
        else
            lr::GUI( translate.getContext(), argc, argv );

        //
        // Clean up
        //
        delete [] text;

    }
    catch( std::exception& e )
    {
        std::cout << "Parsing failed: " << e.what() << std::endl;
        return 0;
    }
}


