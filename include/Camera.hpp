
#ifndef LEGION_CAMERA_H_
#define LEGION_CAMERA_H_


namespace legion
{


class ICamera
{
public:
    struct Sample
    {
        Vector2 pixel;
        Vector2 lens;
        float  time;
    };

    enum PixelFilter
    {
        FILTER_NONE = 0,
        FILTER_BOX,
        FILTER_TENT,
        FILTER_CUBIC_SPLINE
    };

    Camera( const std::string& name );
    virtual ~Camera();

    void setFilter( FilterType filter );
    void setTransform( legion::Matrix4x4, float time );
    void generateRay( const Sample& sample, legion::Ray& transformed_ray );

private:
    virtual void generateCameraSpaceRay( const Sample& filtered_sample, legion::Ray& ray )=0;

    class Impl;
    SharedPtr<Impl> m_impl;
};


}
#endif // LEGION_CAMERA_H_
