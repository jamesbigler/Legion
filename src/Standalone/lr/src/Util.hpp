
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

#ifndef LR_UTIL_HPP_
#define LR_UTIL_HPP_

#include <Legion/Legion.hpp>
#include <rapidxml/rapidxml.hpp>
#include <sstream>
#include <stdexcept>

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

namespace lr
{

bool  readFile( const char* filename, char** contents );


//------------------------------------------------------------------------------
//
// Lexical casting
//
//------------------------------------------------------------------------------

template<typename Target, typename Source>
inline Target lexical_cast( const Source& arg )
{
    std::stringstream interpreter;
    Target result;
    if( !( interpreter << arg     )       ||
        !( interpreter >> result  )       ||
        !( interpreter >> std::ws ).eof() )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
inline legion::Index2 lexical_cast<legion::Index2, std::string>( 
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Index2 result;
    if( !( oss >> result[0] >> result[1] ) ||
        !( oss >> std::ws ).eof()        )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
inline legion::Vector2 lexical_cast<legion::Vector2, std::string>( 
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Vector2 result;
    if( !( oss >> result[0] >> result[1] ) ||
        !( oss >> std::ws ).eof()        )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
inline legion::Vector3 lexical_cast<legion::Vector3, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Vector3 result;
    if( !( oss >> result[0] >> result[1] >> result[2] ) ||
        !( oss >> std::ws ).eof()                     )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
inline legion::Vector4 lexical_cast<legion::Vector4, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Vector4 result;
    if( !( oss >> result[0] >> result[1] >> result[2] >> result[3] ) ||
        !( oss >> std::ws ).eof()                                  )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
inline legion::Matrix lexical_cast<legion::Matrix, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Matrix result;
    if( !(oss >> result[ 0] >> result[ 1] >> result[ 2] >> result[ 3]
              >> result[ 4] >> result[ 5] >> result[ 6] >> result[ 7]
              >> result[ 8] >> result[ 9] >> result[10] >> result[11]
              >> result[12] >> result[13] >> result[14] >> result[15] ) ||
        !( oss >> std::ws ).eof()                                     )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
inline legion::Color lexical_cast<legion::Color, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Color result;
    if( !( oss >> result[0] >> result[1] >> result[2] ) ||
        !( oss >> std::ws ).eof()                     )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}

}

#endif //  LR_UTIL_HPP_
