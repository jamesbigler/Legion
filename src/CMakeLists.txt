
#TODO: Move plugins into separate library and load dynamically

find_package(OptiX REQUIRED)
find_package(CUDA 4.0 REQUIRED)
find_package(CUDA 4.0 REQUIRED)
find_package(OPENEXR REQUIRED)

add_definitions( -DLLOG_MAX_LEVEL=legion::Log::DEBUG )

# Add the path to the OptiX headers to our include paths.
include_directories(
    "${OptiX_INCLUDE}"
    "${CUDA_INCLUDE_DIRS}" 
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${OPENEXR_INCLUDE_PATHS}"
    )

# Tell clang_complete about our include paths
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/clang_complete.in"  "${CMAKE_CURRENT_SOURCE_DIR}/.clang_complete" )


add_subdirectory( Legion)
add_subdirectory( Standalone )
add_subdirectory( Test )

