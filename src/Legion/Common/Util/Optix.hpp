
#ifndef LEGION_COMMON_UTIL_HPP_
#define LEGION_COMMON_UTIL_HPP_

#include <optixu/optixpp_namespace.h>
#include <Legion/Common/Util/Noncopyable.hpp>


namespace legion
{

class Optix : Noncopyable
{
public:
    typedef std::vector<std::string>  ProgramSearchPath;
    Optix();
    ~Optix();

    optix::Context getContext();

    void setProgramSearchPath( const ProgramSearchPath& search_path );

    optix::Program loadProgram( const std::string& filename,
                                const std::string& program_name )const;

private:

    ProgramSearchPath m_search_path;
    optix::Context    m_context;
};

}

#endif // LEGION_COMMON_UTIL_HPP_
