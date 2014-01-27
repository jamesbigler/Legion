

set( PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH
     "Path to where generated PTX is stored" 
     ) 

set( CUDA_GENERATED_OUTPUT_DIR ${PTX_OUTPUT_DIR} )

# Set up build .cu -> .ptx build 
set( CUDA_NVCC_FLAGS "-arch sm_30 --use_fast_math" )
CUDA_COMPILE_PTX( ptx_files
    Objects/Camera/CUDA/ThinLens.cu
    Objects/Environment/CUDA/ConstantEnvironment.cu
    Objects/Environment/CUDA/Environment.cu
    Objects/Geometry/CUDA/Parallelogram.cu
    Objects/Geometry/CUDA/Sphere.cu
    Objects/Geometry/CUDA/TriMesh.cu
    Objects/Light/CUDA/Light.cu
    Objects/Renderer/CUDA/ProgressiveRenderer.cu
    Objects/Surface/CUDA/Beckmann.cu
    Objects/Surface/CUDA/Dielectric.cu
    Objects/Surface/CUDA/DiffuseEmitter.cu
    Objects/Surface/CUDA/Glossy.cu
    Objects/Surface/CUDA/Lambertian.cu
    Objects/Surface/CUDA/Metal.cu
    Objects/Surface/CUDA/Mixture.cu
    Objects/Surface/CUDA/Surface.cu
    Objects/Surface/CUDA/Ward.cu
    Objects/Texture/CUDA/CheckerTexture.cu
    Objects/Texture/CUDA/FresnelSchlickTexture.cu
    Objects/Texture/CUDA/PerlinTexture.cu
    Objects/Texture/CUDA/Texture.cu
    )

configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/Core/config.hpp.in"
                "${CMAKE_BINARY_DIR}/include/config.hpp" )
include_directories( 
    "${CMAKE_BINARY_DIR}/include"
    )

if( WIN32 )
  include_directories(
    "${CMAKE_SOURCE_DIR}/src/Support/openexr-2.0.1/config.windows"
    "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/config.windows"
    )
else()
  include_directories(
    "${CMAKE_SOURCE_DIR}/src/Support/openexr-2.0.1/config"
    "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/config"
    )
endif()

include_directories(
    "${CMAKE_SOURCE_DIR}/src/Support/openexr-2.0.1"
    "${CMAKE_SOURCE_DIR}/src/Support/openexr-2.0.1/IlmImf"
    "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1"
    "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/Half"
    "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/Iex"
    "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/Imath"
    "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/IlmThread"
    )

source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
  
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
    Common/Util/Factory.cpp
    Common/Util/Factory.hpp
    Common/Util/Image.cpp
    Common/Util/Image.hpp
    Common/Util/Logger.cpp
    Common/Util/Logger.hpp
    Common/Util/Noncopyable.hpp
    Common/Util/Parameters.cpp
    Common/Util/Parameters.hpp
    Common/Util/Preprocessor.hpp
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
    Core/Exception.cpp
    Core/Exception.hpp
    Core/PluginContext.cpp
    Core/PluginContext.hpp
    Legion.hpp
    Objects/Camera/ICamera.hpp
    Objects/Camera/ThinLens.cpp
    Objects/Camera/ThinLens.hpp
    Objects/Display/IDisplay.hpp
    Objects/Display/ImageFileDisplay.cpp
    Objects/Display/ImageFileDisplay.hpp
    Objects/Environment/ConstantEnvironment.cpp
    Objects/Environment/ConstantEnvironment.hpp
    Objects/Environment/IEnvironment.hpp
    Objects/Geometry/IGeometry.hpp
    Objects/Geometry/Parallelogram.cpp
    Objects/Geometry/Parallelogram.hpp
    Objects/Geometry/Sphere.cpp
    Objects/Geometry/Sphere.hpp
    Objects/Geometry/TriMesh.cpp
    Objects/Geometry/TriMesh.hpp
    Objects/IObject.cpp
    Objects/IObject.hpp
    Objects/Light/CUDA/Light.hpp
    Objects/Light/ILight.hpp
    Objects/Light/PointLight.cpp
    Objects/Light/PointLight.hpp
    Objects/Renderer/IRenderer.hpp
    Objects/Renderer/ProgressiveRenderer.cpp
    Objects/Renderer/ProgressiveRenderer.hpp
    Objects/Surface/Beckmann.cpp
    Objects/Surface/Beckmann.hpp
    Objects/Surface/Dielectric.cpp
    Objects/Surface/Dielectric.hpp
    Objects/Surface/DiffuseEmitter.cpp
    Objects/Surface/DiffuseEmitter.hpp
    Objects/Surface/Glossy.cpp
    Objects/Surface/Glossy.hpp
    Objects/Surface/ISurface.hpp
    Objects/Surface/Lambertian.cpp
    Objects/Surface/Lambertian.hpp
    Objects/Surface/Metal.cpp
    Objects/Surface/Metal.hpp
    Objects/Surface/Mixture.cpp
    Objects/Surface/Mixture.hpp
    Objects/Surface/Ward.cpp
    Objects/Surface/Ward.hpp
    Objects/Texture/CheckerTexture.cpp
    Objects/Texture/CheckerTexture.hpp
    Objects/Texture/ConstantTexture.cpp
    Objects/Texture/ConstantTexture.hpp
    Objects/Texture/FresnelSchlickTexture.cpp
    Objects/Texture/FresnelSchlickTexture.hpp
    Objects/Texture/ITexture.cpp
    Objects/Texture/ITexture.hpp
    Objects/Texture/ImageTexture.cpp
    Objects/Texture/ImageTexture.hpp
    Objects/Texture/PerlinTexture.cpp
    Objects/Texture/PerlinTexture.hpp
    Objects/VariableContainer.cpp
    Objects/VariableContainer.hpp
    Renderer/OptiXProgramManager.cpp
    Renderer/OptiXProgramManager.hpp
    Renderer/OptiXScene.cpp
    Renderer/OptiXScene.hpp
    )

# Link against optix
target_link_libraries( legion
    ${Boost_LIBRARIES}
    Half
    Iex
    IexMath
    IlmImf
    IlmThread
    Imath
    optix
    ${optix_rpath}
    zlib
    )

if (WIN32)
  add_custom_command(OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(Configuration)/optix.1.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
    "${optix_DLL}"
     "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(Configuration)"
     DEPENDS "${optix_DLL}"
     COMMENT "${CMAKE_COMMAND} -E copy ${optix_DLL} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(Configuration)"
     )
  add_custom_target(optix_copy_dlls
    DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(Configuration)/optix.1.dll"
    DEPENDS "${optix_DLL}"
    )
  # Add optix_copy_dlls target as a dependency of legion
  add_dependencies(legion optix_copy_dlls)
endif()


