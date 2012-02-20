
#TODO: Move plugins into separate library and load dynamically

find_package(OptiX REQUIRED)
find_package(CUDA 4.0 REQUIRED)

add_definitions( -DLLOG_MAX_LEVEL=legion::Log::DEBUG )

# Add the path to the OptiX headers to our include paths.
include_directories(
    "${OptiX_INCLUDE}"
    "${CUDA_INCLUDE_DIRS}" 
    "${CMAKE_CURRENT_SOURCE_DIR}"
    )

add_subdirectory( Legion)
add_subdirectory( Standalone )
add_subdirectory( Test )

