
#TODO: Move plugins into separate library and load dynamically

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE "Release" CACHE STRING
         "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
         FORCE )
endif( NOT CMAKE_BUILD_TYPE )

add_definitions( -DLLOG_MAX_LEVEL=legion::Log::DEBUG )

set(CMAKE_CXX_FLAGS                "-Wall -Werror -funroll-loops -floop-optimize -ffast-math -Qunused-arguments -stdlib=libc++" )
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

find_package( OptiX REQUIRED )
find_package( CUDA 4.0 REQUIRED )
find_package( CUDA 4.0 REQUIRED )
find_package( OPENEXR REQUIRED )
find_package( Boost COMPONENTS thread REQUIRED )


# Add the path to the OptiX headers to our include paths.
include_directories(
    "${Boost_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CUDA_INCLUDE_DIRS}" 
    "${OPENEXR_INCLUDE_PATHS}"
    "${OptiX_INCLUDE}"
    )

# Tell clang_complete about our include paths
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/clang_complete.in"  "${CMAKE_CURRENT_SOURCE_DIR}/.clang_complete" )


add_subdirectory( Legion)
add_subdirectory( Standalone )
add_subdirectory( Test )

