

# Set up build .cu -> .ptx build 
CUDA_COMPILE_PTX( ptx_files
    Cuda/hit_programs.cu
    Cuda/ray_generation.cu
    Cuda/triangle_mesh.cu
    )


# build the library and ptx files
add_library( legion SHARED
    ${ptx_files}
    Common/Math/Filter.hpp
    Common/Math/Math.cpp
    Common/Math/Math.hpp
    Common/Util/Assert.hpp
    Common/Util/Logger.cpp
    Common/Util/Logger.hpp
    Common/Util/Noncopyable.hpp
    Common/Util/Singleton.hpp
    Common/Util/Optix.cpp
    Common/Util/Optix.hpp
    Common/Util/Stream.cpp
    Common/Util/Stream.hpp
    Core/APIBase.cpp
    Core/APIBase.hpp
    Core/Color.hpp
    Core/Context.cpp
    Core/Context.hpp
    Core/ContextImpl.cpp
    Core/ContextImpl.hpp
    Core/Exception.hpp
    Core/Light.hpp
    Core/Matrix.hpp
    Core/Ray.hpp
    Core/Vector.hpp
    Cuda/ray_generation.cu
    Cuda/triangle_mesh.cu
    Legion.hpp
    Scene/Camera/BasicCamera.cpp
    Scene/Camera/BasicCamera.hpp
    Scene/Camera/ICamera.cpp
    Scene/Camera/ICamera.hpp
    Scene/Camera/ThinLensCamera.cpp
    Scene/Camera/ThinLensCamera.hpp
    Scene/Film/Film.hpp
    Scene/Film/IFilm.cpp
    Scene/Film/IFilm.hpp
    Scene/Film/ImageFilm.cpp
    Scene/Film/ImageFilm.hpp
    Scene/LightShader/ILightShader.cpp
    Scene/LightShader/ILightShader.hpp
    Scene/LightShader/PointLightShader.cpp
    Scene/LightShader/PointLightshader.hpp
    Scene/Mesh/Mesh.cpp
    Scene/Mesh/Mesh.hpp
    Scene/SurfaceShader/ISurfaceShader.cpp
    Scene/SurfaceShader/ISurfaceShader.hpp
    Scene/SurfaceShader/LambertianShader.cpp
    Scene/SurfaceShader/LambertianShader.hpp
    )

  # Link against optix
target_link_libraries( legion
    optix
    ${optix_rpath}
    )



