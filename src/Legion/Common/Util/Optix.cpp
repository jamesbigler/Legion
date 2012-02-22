
#include <Legion/Common/Util/Optix.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Core/Exception.hpp>

using namespace legion;

namespace
{
}


Optix::Optix()
    : m_context( optix::Context::create() )
{
}


Optix::~Optix()
{
    m_context->destroy();
}


optix::Context Optix::getContext()
{
    return m_context;
}


void Optix::setProgramSearchPath( const std::string& search_path )
{
    m_search_path = search_path;
}


void Optix::registerProgram( const std::string& filename,
                             const std::string& program_name )
{
    std::string path;
    if( !m_search_path.empty() )  path = m_search_path + "/" + filename;
    else                          path = filename;

    try
    {
        optix::Program program;
        program = m_context->createProgramFromPTXFile( path, program_name );
        m_program_map.insert( std::make_pair( program_name, program ) );

        LLOG_INFO << "Successfully loaded optix program " << program_name;
        LLOG_INFO << "    from: " << filename;
    }
    OPTIX_CATCH_RETHROW;
}
    

optix::Program Optix::getProgram( const std::string& program_name )const
{
    ProgramMap::const_iterator program = m_program_map.find( program_name );
    if( program == m_program_map.end() )
    {
        throw Exception( "Couldnt retrieve optix program!" );
    }
    return program->second;
}

