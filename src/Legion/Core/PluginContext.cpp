
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

#include <Legion/Core/PluginContext.hpp>
#include <Legion/Core/Exception.hpp>
#include <fstream>

using namespace legion;

PluginContext::PluginContext( optix::Context optix_context ) 
    : m_optix_context( optix_context )
{
}


void PluginContext::addAssetPath( const std::string& asset_path )
{
    m_asset_path.push_back( asset_path );
}


optix::Context PluginContext::getOptiXContext()
{
    return m_optix_context;
}







#include <iostream>





void PluginContext::openFile(
        const std::string& filename,
        std::ifstream& out,
        std::ios_base::openmode mode
        ) const
{
    for( Path::const_iterator it = m_asset_path.begin();
         it != m_asset_path.end();
         ++it )
    {
        out.open( (*it + "/" + filename).c_str(), mode );
        if( out.is_open() )
            return;
        std::cout << "failed to open '" << (*it + "/" + filename) << "'" << std::endl;
    }
    throw Exception( "Failed to open file '" + filename + "' for reading" );
}
