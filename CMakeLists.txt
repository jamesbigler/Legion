
cmake_minimum_required( VERSION 2.8 )
project( Legion ) 

include_directories( "${CMAKE_SOURCE_DIR}/include" )

add_subdirectory( src )
add_subdirectory( standalone )
