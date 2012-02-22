
#ifndef LEGION_COMMON_UTIL_HPP_
#define LEGION_COMMON_UTIL_HPP_

#include <optixu/optixpp_namespace.h>
#include <Legion/Common/Util/Noncopyable.hpp>
#include <map>


#define OPTIX_CATCH_RETHROW                                                    \
    catch ( optix::Exception& e )                                              \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: ")+e.what() );  \
    }                                                                          \
    catch ( std::exception& e )                                                \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: ")+e.what() );  \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
        throw legion::Exception( std::string("OPTIX_EXCEPTION: unknown") );    \
    }

#define OPTIX_CATCH_WARN                                                       \
    catch ( optix::Exception& e )                                              \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: " << e.what();                          \
    }                                                                          \
    catch ( std::exception& e )                                                \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: " << e.what();                          \
    }                                                                          \
    catch (...)                                                                \
    {                                                                          \
        LLOG_WARN << "OPTIX_EXCEPTION: Unknown";                               \
    }

namespace legion
{

class Optix : Noncopyable
{
public:
    typedef std::map<std::string, optix::Program>  ProgramMap;

    Optix();
    ~Optix();

    optix::Context getContext();

    void setProgramSearchPath( const std::string& path );

    void registerProgram( const std::string& filename,
                          const std::string& program_name );

    optix::Program getProgram( const std::string& program_name )const;
private:

    optix::Program loadProgram( const std::string& filename,
                                const std::string& program_name );

    optix::Context     m_context;
    std::string        m_search_path;
    ProgramMap         m_program_map;
};

}

#endif // LEGION_COMMON_UTIL_HPP_
