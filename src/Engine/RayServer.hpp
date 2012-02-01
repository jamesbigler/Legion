
#ifndef LEGION_ENGINE_RAY_SERVER_HPP_
#define LEGION_ENGINE_RAY_SERVER_HPP_

namespace legion
{
    class RayServer
    {
    public:
        RayServer() {}
        
        unsigned addRays( unsigned num_rays, const Rays* rays ); // Returns beginning index 

    }
}

#endif // LEGION_ENGINE_RAY_SERVER_HPP_
