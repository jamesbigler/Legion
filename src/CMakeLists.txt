
#TODO: Move plugins into separate library and load dynamically

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
    "${OptiX_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    )

# Set up build .cu -> .ptx build 
find_package(CUDA 4.0 REQUIRED)
CUDA_COMPILE_PTX( ptx_files
    Cuda/ray_generation.cu
    Cuda/triangle_mesh.cu
    )

# build the library and ptx files
add_library( legion SHARED
    ${ptx_files}
    ../include/Core/APIBase.hpp
    ../include/Core/Color.hpp
    ../include/Core/Context.hpp
    ../include/Core/Exception.hpp
    ../include/Core/Matrix.hpp
    ../include/Core/Mesh.hpp
    ../include/Core/Ray.hpp
    ../include/Core/Vector.hpp
    ../include/Interface/ICamera.hpp
    ../include/Interface/IFilm.hpp
    ../include/Interface/ILightShader.hpp
    ../include/Interface/ISurfaceShader.hpp
    ../include/Util/InternalHelpers.hpp
    ../include/Util/Math.hpp
    ../include/Util/Stream.hpp
    ../include/Util/Util.hpp
    Common/Math/Math.cpp
    Common/Util/Stream.cpp
    Core/Context.cpp
    Core/ContextImpl.cpp
    Core/ContextImpl.hpp
    Core/APIBase.cpp
    Scene/Camera/BasicCamera.cpp
    Scene/Camera/BasicCamera.hpp
    Scene/Camera/ICamera.cpp
    Scene/Camera/ThinLensCamera.cpp
    Scene/Camera/ThinLensCamera.hpp
    Scene/Film/Film.hpp
    Scene/Film/IFilm.cpp
    Scene/Film/ImageFilm.cpp
    Scene/Film/ImageFilm.hpp
    Scene/LightShader/ILightShader.cpp
    Scene/LightShader/PointLightShader.cpp
    Scene/LightShader/PointLightshader.hpp
    Scene/Mesh/Mesh.cpp
    Scene/SurfaceShader/ISurfaceShader.cpp
    Scene/SurfaceShader/LambertianShader.cpp
    Scene/SurfaceShader/LambertianShader.hpp
    )

# Link against optix
target_link_libraries( legion
    optix
    ${optix_rpath}
    )



