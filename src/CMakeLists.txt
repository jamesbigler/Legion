
#TODO: Move plugins into separate library and load dynamically

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
    "${OptiX_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    )

add_subdirectory( Legion)
add_subdirectory( Standalone )
add_subdirectory( Test )

