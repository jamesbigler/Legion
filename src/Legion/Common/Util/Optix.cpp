
#include <Legion/Common/Util/Optix.hpp>
#include <Legion/Common/Util/Logger.hpp>

using namespace legion;


optix::Context Optix::s_context;


optix::Context Optix::context()
{
    if( !optix::Context() )
    {
        LLOG_INFO << "Creating optix::Context";
        std::atexit( destroyContext );
        s_context = optix::Context::create();
    }

    return s_context;
}


void Optix::destroyContext()
{
    LLOG_INFO << "Destroying optix::Context";
    s_context->destroy(); 
}

