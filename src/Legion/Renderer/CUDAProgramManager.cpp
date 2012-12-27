
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

#include <Legion/Renderer/CUDAProgramManager.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Core/Exception.hpp>
#include <sstream>

using namespace legion;


CUDAProgramManager::CUDAProgramManager( optix::Context context )
    : m_context( context )
{
}


CUDAProgramManager::~CUDAProgramManager()
{
}


void CUDAProgramManager::addPath( const std::string& path )
{
    m_paths.push_back( path );
}


optix::Program CUDAProgramManager::get( 
        const std::string& name,
        const std::string& cuda_filename,
        const std::string& cuda_function_name )
{
    
    const std::string lookup_name   = cuda_filename + cuda_function_name + name;
    Registry::iterator it = m_registry.find( lookup_name );
    if( it != m_registry.end() )
    {
        return it->second;
    }

    const std::string filename      = cuda_filename.empty() ? 
                                      name + ".cu"          :
                                      cuda_filename;

    const std::string function_name = cuda_function_name.empty() ?
                                      name                       :
                                      cuda_function_name;
                           
    for( Paths::iterator path = m_paths.begin(); path != m_paths.end(); ++path )
    {
        std::string full_path = *path + "/" + filename; 
        try
        {
            optix::Program p = 
                m_context->createProgramFromPTXFile( full_path, function_name );
            if( p )
            {
                m_registry.insert( std::make_pair( lookup_name, p ) );
                return p;
            }
        }
        catch( optix::Exception& e )
        {
            if( e.getErrorCode() == RT_ERROR_FILE_NOT_FOUND ) continue;

            LLOG_INFO << "OptiX exception '" 
                      << m_context->getErrorString( e.getErrorCode() );
            throw;
        }
        catch( std::exception& e )
        {
        }
        catch( ... )
        {
            LLOG_ERROR << "Unknown exception thrown by optix::Context::"
                       << "createProgramFromPTXFile\n"
                       << "\tname: " << name << "\n"
                       << "\tfile: " << cuda_filename << "\n"
                       << "\tfunc: " << cuda_function_name;
            throw;
        }

    }
   
    std::ostringstream iss;
    iss << "Failed to create program '" << name << "' from function '"
        << cuda_function_name << "' in file '" << cuda_filename << "'";
    throw Exception( iss.str() ); 
}

