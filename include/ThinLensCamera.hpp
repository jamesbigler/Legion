
#ifndef LEGION_CAMERA_H_
#define LEGION_CAMERA_H_


// TODO:
//   - Remove non-copyable semantics and make Impl* member be shared_ptr?


namespace legion
{

class Camera
{
public:
    struct Sample
    {
        float2 pixel;
        float2 lens;
        float  time;
    };

    enum Filter
    {
        FILTER_BOX = 0,
        FILTER_TENT,
        FILTER_CUBIC_SPLINE
    };

    virtual ~Camera();

    void setViewPlane( float left, float right, float bottom, float top );
    void setShutterOpenClose( float open, float close );
    void setFocalDistance( float distance );
    void setFilter( FilterType filter );
    void setLensRadius( float radius );
    void setTransform( legion::Matrix4x4 );

    virtual void getRay( const Sample& sample, legion::Ray& ray )=0;

private:
    class Impl;
    SharedPtr<Impl> m_impl;
};

}
#endif // LEGION_CAMERA_H_
