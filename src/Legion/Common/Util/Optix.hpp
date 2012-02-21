
#ifndef LEGION_COMMON_UTIL_HPP_
#define LEGION_COMMON_UTIL_HPP_

#include <optixu/optixpp_namespace.h>
#include <Legion/Common/Util/Noncopyable.hpp>
#include <map>


namespace legion
{

class Optix : Noncopyable
{
public:
    typedef std::vector<std::string>               ProgramSearchPath;
    typedef std::map<std::string, optix::Program>  ProgramMap;

    Optix();
    ~Optix();

    optix::Context getContext();

    void setProgramSearchPath( const ProgramSearchPath& search_path );

    void registerProgram( const std::string& filename,
                          const std::string& program_name );

    optix::Program getProgram( const std::string& program_name )const;
private:

    optix::Program loadProgram( const std::string& path,
                                const std::string& filename,
                                const std::string& program_name );

    optix::Context     m_context;

    ProgramSearchPath  m_search_path;
    ProgramMap         m_program_map;
};

}

#endif // LEGION_COMMON_UTIL_HPP_
