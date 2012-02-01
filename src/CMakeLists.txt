
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
    ../include/Camera/ThinLensCamera.hpp
    ../include/Core/Color.hpp
    ../include/Core/Context.hpp
    ../include/Core/Exception.hpp
    ../include/Core/Index.hpp
    ../include/Core/Matrix.hpp
    ../include/Core/Mesh.hpp
    ../include/Core/Ray.hpp
    ../include/Core/Vector.hpp
    ../include/Film/Film.hpp
    ../include/Film/ImageFilm.hpp
    ../include/Interface/IBasicCamera.hpp
    ../include/Interface/ICamera.hpp
    ../include/Interface/IFilm.hpp
    ../include/Interface/ILightShader.hpp
    ../include/Interface/ISurfaceShader.hpp
    ../include/LightShader/PointLightShader.hpp
    ../include/Private/APIBase.hpp
    ../include/SurfaceShader/LambertianShader.hpp
    ../include/Util/Stream.hpp
    ../include/Util/Util.h
    Camera/IBasicCamera.cpp
    Camera/ICamera.cpp
    Camera/ThinLensCamera.cpp
    Core/APIBase.cpp
    Core/Context.cpp
    Engine/ContextImpl.cpp
    Engine/ContextImpl.hpp
    Engine/PrimaryRayGenerator.cpp
    Engine/PrimaryRayGenerator.hpp
    Engine/Light.hpp
    Film/IFilm.cpp
    Film/ImageFilm.cpp
    LightShader/ILightShader.cpp
    LightShader/PointLightShader.cpp
    Math/Math.cpp
    Mesh/Mesh.cpp
    SurfaceShader/ISurfaceShader.cpp
    SurfaceShader/LambertianShader.cpp
    Util/Stream.cpp
    )

# Link against optix
target_link_libraries( legion
    optix
    ${optix_rpath}
    )



