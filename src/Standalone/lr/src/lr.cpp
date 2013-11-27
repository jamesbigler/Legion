
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
    std::cout
        << "\nUsage  : " << argv0 << " [options] <scene_file.xml>\n\n"
        << "\t-n | --num_samples    <NUM_SAMPLES>  Set number of samples\n"
        << "\t-r | --res            <X> <Y>        Set image resolution\n"
        << "\t-s | --specular-depth <SPEC_DEPTH>   Set maximum specular depth\n"
        << "\t-d | --diffuse-depth  <DIFF_DEPTH>   Set maximum diffusedepth\n"
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

    int num_samples = 0;
    int res_x       = 0;
    int res_y       = 0;
    int diff_depth  = -1;
    int spec_depth  = -1;
    for( int i = 1; i < argc-1; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "-n" || arg == "--num-samples" )
        {
            if( i >= argc-2 )
                printUsageAndExit( argv[0] );
            num_samples = lr::lexical_cast<int>( argv[++i] );
        }
        else if( arg == "-r" || arg == "--res" )
        {
            if( i >= argc-3 )
                printUsageAndExit( argv[0] );
            res_x = lr::lexical_cast<int>( argv[++i] );
            res_y = lr::lexical_cast<int>( argv[++i] );
        }
        else if( arg == "-d" || arg == "--diffuse-depth" )
        {
            if( i >= argc-2 )
                printUsageAndExit( argv[0] );
            diff_depth = lr::lexical_cast<int>( argv[++i] );
        }
        else if( arg == "-s" || arg == "--specular-depth" )
        {
            if( i >= argc-2 )
                printUsageAndExit( argv[0] );
            spec_depth = lr::lexical_cast<int>( argv[++i] );
        }
        else
        {
            printUsageAndExit( argv[0] );
        }
    }

    // Render scene using IDisplay specified in xml file
    try
    {
        const std::string scene_file = argv[ argc-1 ];
        char* text;
        if( ! lr::readFile( scene_file.c_str(), &text ) )
            throw std::runtime_error( "Failed to read xml file." );

        const size_t pos = scene_file.find_last_of( '/' );
        const std::string scene_dir = pos == std::string::npos ?
                                      "."                      :
                                      scene_file.substr( 0, pos );
        legion::Context context;
        context.addAssetPath( scene_dir );
        lr::XMLToLegion translate( text, &context, true );
        if( num_samples )
            context.getRenderer()->setSamplesPerPixel( num_samples );
        if( res_x && res_y )
            context.getRenderer()->setResolution(legion::Index2(res_x, res_y));
        if( diff_depth >= 0 )
            context.getRenderer()->setMaxDiffuseDepth( diff_depth );
        if( spec_depth >= 0 )
            context.getRenderer()->setMaxSpecularDepth( spec_depth );
        context.render();

        delete [] text;

    }
    catch( std::exception& e )
    {
        std::cout << "lr failure: " << e.what() << std::endl;
        return 0;
    }
}


