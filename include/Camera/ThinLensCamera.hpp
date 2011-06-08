
#ifndef LEGION_CAMERA_H_
#define LEGION_CAMERA_H_


namespace legion
{


class ThinLensCamera : public IBasicCamera
{
public:
    explicit ThinLensCamera( const std::string& name );
    ~ThinLensCamera();

    void setViewPlane( float left, float right, float bottom, float top );
    void setFocalDistance( float distance );
    void setLensRadius( float radius );

private:
    void generateCameraSpaceRay( const Sample& filtered_sample, CameraSpaceRay& ray );

    class Impl;
    std::shared_ptr<Impl> m_impl;
};


}
#endif // LEGION_CAMERA_H_
