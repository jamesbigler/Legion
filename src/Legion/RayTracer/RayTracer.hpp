
#ifndef LEGION_RAYTRACER_RAYTRACER_HPP_
#define LEGION_RAYTRACER_RAYTRACER_HPP_


namespace legion
{

class RayTracer
{
public:
    RayTracer();

    optix::Buffer createBuffer();

    void updateVertexBuffer( optix::Buffer buffer,
                             unsigned num_verts,
                             const Vertex* verts );

    void updateTriangleBuffer( optix::Buffer buffer,
                               unsigned num_tris,
                               const optix::Index3* tris );









private:
};

}


#endif // LEGION_RAYTRACER_RAYTRACER_HPP_

