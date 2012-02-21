
#include <Legion/Common/Util/Optix.hpp>
#include <Legion/Common/Util/Logger.hpp>

using namespace legion;

namespace
{
    optix::Program loadProgram( optix::Context context,
                                const std::string& path,
                                const std::string& filename,
                                const std::string& program_name )
    {
        std::string full_path;
        if( !path.empty() )
            full_path = path + "/" + filename;
        else
            full_path = filename;

        try
        {
            return context->createProgramFromPTXFile( full_path, program_name );
        }
        catch( ... )
        {
            return optix::Program();

        }
    }
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


optix::Program Optix::loadProgram( const std::string& filename,
                                   const std::string& program_name )const
{

    for( ProgramSearchPath::const_iterator path = m_search_path.begin();
         path != m_search_path.end();
         ++path )
    {
        optix::Program program = ::loadProgram( m_context,
                                              *path,
                                              filename,
                                              program_name );
        if( program )
        {
            LLOG_INFO << "Successfully loaded optix program " << program_name;
            return program;
        }
    }
    LLOG_INFO << "Failed to load optix program " << program_name;
    
    throw "Couldnt load program!!";

}
