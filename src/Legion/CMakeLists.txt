
set( PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH
     "Path to where generated PTX is stored" 
     ) 

set( CUDA_GENERATED_OUTPUT_DIR ${PTX_OUTPUT_DIR} )


# Set up build .cu -> .ptx build 
CUDA_COMPILE_PTX( ptx_files
    Renderer/Cuda/hit.cu
    Renderer/Cuda/raygen.cu
    Renderer/Cuda/trimesh.cu
    )

configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/Core/config.hpp.in"
                "${CMAKE_BINARY_DIR}/include/config.hpp" )
include_directories( "${CMAKE_BINARY_DIR}/include" )


# build the library and ptx files
add_library( legion SHARED
    ${ptx_files}
    Common/Math/Filter.hpp
    Common/Math/MTRand.cpp
    Common/Math/MTRand.hpp
    Common/Math/Math.cpp
    Common/Math/Math.hpp
    Common/Math/Matrix.hpp
    Common/Math/ONB.hpp
    Common/Math/Sobol.cpp
    Common/Math/Sobol.hpp
    Common/Math/Vector.hpp
    Common/Util/Assert.hpp
    Common/Util/AutoTimerHelpers.hpp
    Common/Util/Image.cpp
    Common/Util/Image.hpp
    Common/Util/Logger.cpp
    Common/Util/Logger.hpp
    Common/Util/Noncopyable.hpp
    Common/Util/Singleton.hpp
    Common/Util/Stream.cpp
    Common/Util/Stream.hpp
    Common/Util/Timer.cpp
    Common/Util/Timer.hpp
    Common/Util/TypeConversion.cpp
    Common/Util/TypeConversion.hpp
    Common/Util/Util.hpp
    Core/APIBase.cpp
    Core/APIBase.hpp
    Core/Color.hpp
    Core/Context.cpp
    Core/Context.hpp
    Core/Declarations.hpp
    Core/Exception.cpp
    Core/Exception.hpp
    Core/Ray.hpp
    Legion.hpp
    Renderer/LightSet.cpp
    Renderer/LightSet.hpp
    Renderer/RayScheduler.cpp
    Renderer/RayScheduler.hpp
    Renderer/RayServer.hpp
    Renderer/RayTracer.cpp
    Renderer/RayTracer.hpp
    Renderer/Renderer.cpp
    Renderer/Renderer.hpp
    Renderer/ShadingEngine.cpp
    Renderer/ShadingEngine.hpp
    Scene/Camera/IBasicCamera.cpp
    Scene/Camera/IBasicCamera.hpp
    Scene/Camera/ICamera.cpp
    Scene/Camera/ICamera.hpp
    Scene/Camera/ThinLensCamera.cpp
    Scene/Camera/ThinLensCamera.hpp
    Scene/Film/Film.hpp
    Scene/Film/IFilm.cpp
    Scene/Film/IFilm.hpp
    Scene/Film/ImageFilm.cpp
    Scene/Film/ImageFilm.hpp
    Scene/LightShader/DiffuseLight.cpp
    Scene/LightShader/DiffuseLight.hpp
    Scene/LightShader/ILightShader.cpp
    Scene/LightShader/ILightShader.hpp
    Scene/LightShader/MeshLight.cpp
    Scene/LightShader/MeshLight.hpp
    Scene/LightShader/PointLightShader.cpp
    Scene/LightShader/PointLightShader.hpp
    Scene/Mesh/Mesh.cpp
    Scene/Mesh/Mesh.hpp
    Scene/SurfaceShader/ISurfaceShader.cpp
    Scene/SurfaceShader/ISurfaceShader.hpp
    Scene/SurfaceShader/Lambertian.cpp
    Scene/SurfaceShader/Lambertian.hpp
    )

# Link against optix
target_link_libraries( legion
    optix
    ${optix_rpath}
    ${OPENEXR_LIBRARIES}
    ${Boost_LIBRARIES}
    )



