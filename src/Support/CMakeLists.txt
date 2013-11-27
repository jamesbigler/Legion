
if(WIN32)
  # /wd4996 - strncpy and other functions are unsafe
  # /wd4800 - forcing value to bool 'true' or 'false' (performance warning)
  # C4267: 'return' : conversion from 'size_t' to 'int', possible loss of data
  # C4244: 'argument' : conversion from 'ptrdiff_t' to 'int', possible loss of data
  # C4305: 'argument' : truncation from 'double' to 'float'
  # C4334: '<<' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)
  add_definitions(/wd4996 /wd4800 /wd4267 /wd4244 /wd4305 /wd4334)
endif()

set( cache_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" )
add_subdirectory( ilmbase-2.0.1 )
add_subdirectory( openexr-2.0.1 )
add_subdirectory( zlib-1.2.8 )
set( CMAKE_CXX_FLAGS ${cache_CMAKE_CXX_FLAGS} )
