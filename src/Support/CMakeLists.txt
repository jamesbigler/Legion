

set( cache_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" )
add_subdirectory( ilmbase-2.0.1 )
add_subdirectory( openexr-2.0.1 )
add_subdirectory( zlib-1.2.8 )
set( CMAKE_CXX_FLAGS ${cache_CMAKE_CXX_FLAGS} )
