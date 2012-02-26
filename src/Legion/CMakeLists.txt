
set( PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH "Path to where generated PTX is stored" ) 
set( CUDA_GENERATED_OUTPUT_DIR ${PTX_OUTPUT_DIR} )


# Set up build .cu -> .ptx build 
CUDA_COMPILE_PTX( ptx_files
    Renderer/Cuda/hit.cu
    Renderer/Cuda/raygen.cu
    Renderer/Cuda/trimesh.cu
    )

configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/Core/config.hpp.in"  "${CMAKE_CURRENT_SOURCE_DIR}/Core/config.hpp" )

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
    Common/Util/Stream.cpp
    Common/Util/Stream.hpp
    Core/APIBase.cpp
    Core/APIBase.hpp
    Core/Color.hpp
    Core/Context.cpp
    Core/Context.hpp
    Core/Exception.cpp
    Core/Exception.hpp
    Core/Light.hpp
    Core/Matrix.hpp
    Core/Ray.hpp
    Core/Vector.hpp
    Legion.hpp
    Renderer/Renderer.cpp
    Renderer/Renderer.hpp
    Renderer/RayScheduler.cpp
    Renderer/RayScheduler.hpp
    Renderer/RayQueue.cpp
    Renderer/RayQueue.hpp
    Renderer/RayTracer.cpp
    Renderer/RayTracer.hpp
    Renderer/ShadingEngine.cpp
    Renderer/ShadingEngine.hpp
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
    Scene/LightShader/PointLightShader.hpp
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


