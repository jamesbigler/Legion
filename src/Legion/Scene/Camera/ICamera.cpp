
#include <Legion/Scene/Camera/ICamera.hpp>

using namespace legion;


ICamera::ICamera( Context* context, const std::string& name )
    : APIBase( context, name )
{
}


ICamera::~ICamera()
{
}
