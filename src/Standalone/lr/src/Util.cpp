
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


#include <Util.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace lr;


bool lr::readFile( const char* filename, char** contents )
{
    std::ifstream in( filename, std::ios::binary );
    if( !in )
    {
        std::cout << "Failed to open file '" << filename << "'"
                  << std::endl;
        return false;
    }

    // get length of file:
    in.seekg( 0, std::ios::end );
    size_t length = in.tellg();
    in.seekg (0, std::ios::beg );

    // read data 
    *contents = new char[ length+1 ];
    in.read( *contents, length );
    (*contents)[length] = '\0';

    return true;
}
