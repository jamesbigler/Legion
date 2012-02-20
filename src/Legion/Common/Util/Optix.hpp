
#ifndef LEGION_COMMON_UTIL_HPP_
#define LEGION_COMMON_UTIL_HPP_

#include <optixu/optixpp_namespace.h>


namespace legion
{

class Optix
{
public:
    static optix::Context context();

private:
    static void destroyContext();

    static optix::Context s_context;
};

}

#endif // LEGION_COMMON_UTIL_HPP_
