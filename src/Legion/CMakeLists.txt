
set( PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH
     "Path to where generated PTX is stored" 
     ) 

set( CUDA_GENERATED_OUTPUT_DIR ${PTX_OUTPUT_DIR} )


# Set up build .cu -> .ptx build 
set( CUDA_NVCC_FLAGS "-arch sm_30" )
CUDA_COMPILE_PTX( ptx_files
    Objects/Camera/CUDA/ThinLens.cu
    Objects/Geometry/CUDA/Sphere.cu
    Objects/Renderer/CUDA/ProgressiveRenderer.cu
    Objects/Surface/CUDA/Lambertian.cu
    Objects/Surface/CUDA/Normal.cu
    Objects/Surface/CUDA/Surface.cu
    )

configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/Core/config.hpp.in"
                "${CMAKE_BINARY_DIR}/include/config.hpp" )
include_directories( 
    "${CMAKE_BINARY_DIR}/include"
    )


# build the library and ptx files
add_library( legion SHARED
    ${ptx_files}
    Common/Math/CUDA/Math.hpp
    Common/Math/CUDA/ONB.hpp
    Common/Math/CUDA/Sobol.hpp
    Common/Math/Filter.hpp
    Common/Math/MTRand.cpp
    Common/Math/MTRand.hpp
    Common/Math/Math.hpp
    Common/Math/Matrix.hpp
    Common/Math/Vector.hpp
    Common/Util/Assert.hpp
    Common/Util/AutoTimerHelpers.hpp
    Common/Util/Image.cpp
    Common/Util/Image.hpp
    Common/Util/Logger.cpp
    Common/Util/Logger.hpp
    Common/Util/Noncopyable.hpp
    Common/Util/Parameters.cpp
    Common/Util/Parameters.hpp
    Common/Util/Plugin.cpp
    Common/Util/Plugin.hpp
    Common/Util/Singleton.hpp
    Common/Util/Stream.cpp
    Common/Util/Stream.hpp
    Common/Util/Timer.cpp
    Common/Util/Timer.hpp
    Common/Util/TypeConversion.cpp
    Common/Util/TypeConversion.hpp
    Common/Util/Util.hpp
    Core/Color.hpp
    Core/Context.cpp
    Core/Context.hpp
    Core/Declarations.hpp
    Core/Exception.cpp
    Core/Exception.hpp
    Core/Ray.hpp
    Core/VariableContainer.cpp
    Core/VariableContainer.hpp
    Legion.hpp
    Objects/Camera/ICamera.hpp
    Objects/Camera/ThinLens.cpp
    Objects/Camera/ThinLens.hpp
    Objects/Display/IDisplay.hpp
    Objects/Display/ImageFileDisplay.cpp
    Objects/Display/ImageFileDisplay.hpp
    Objects/Geometry/IGeometry.hpp
    Objects/Geometry/Sphere.cpp
    Objects/Geometry/Sphere.hpp
    Objects/Geometry/TriMesh.cpp
    Objects/Geometry/TriMesh.hpp
    Objects/IObject.cpp
    Objects/IObject.hpp
    Objects/Light/ILight.hpp
    Objects/Light/PointLight.cpp
    Objects/Light/PointLight.hpp
    Objects/Renderer/IRenderer.hpp
    Objects/Renderer/ProgressiveRenderer.cpp
    Objects/Renderer/ProgressiveRenderer.hpp
    Objects/Surface/ISurface.hpp
    Objects/Surface/Lambertian.cpp
    Objects/Surface/Lambertian.hpp
    Renderer/CUDAProgramManager.cpp
    Renderer/CUDAProgramManager.hpp
    Renderer/OptiXScene.cpp
    Renderer/OptiXScene.hpp
    )

# Link against optix
target_link_libraries( legion
    optix
    ${optix_rpath}
    ${OPENEXR_LIBRARIES}
    ${Boost_LIBRARIES}
    )



