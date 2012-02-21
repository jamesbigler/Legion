
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


void Optix::setProgramSearchPath( const ProgramSearchPath& search_path )
{
    m_search_path.assign( search_path.begin(), search_path.end() );
}


optix::Program Optix::loadProgram( const std::string& path,
                                   const std::string& filename,
                                   const std::string& program_name )
{
    std::string full_path;
    if( !path.empty() )  full_path = path + "/" + filename;
    else                 full_path = filename;

    try
    {
        return m_context->createProgramFromPTXFile( full_path, program_name );
    }
    catch( ... )
    {
        return optix::Program();
    }
}


void Optix::registerProgram( const std::string& filename,
                             const std::string& program_name )
{

    for( ProgramSearchPath::const_iterator path = m_search_path.begin();
         path != m_search_path.end();
         ++path )
    {
        optix::Program program = loadProgram( *path, filename, program_name );
        if( program )
        {
            LLOG_INFO << "Successfully loaded optix program " << program_name;
            LLOG_INFO << "    from: " << filename;

            m_program_map.insert( std::make_pair( program_name, program ) );
            return;
        }
    }

    LLOG_INFO << "Failed to load optix program '" << program_name;
    LLOG_INFO << "   from: " << filename;
    throw Exception( "Couldnt load optix program!!" );

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

