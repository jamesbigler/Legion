
#TODO: Move plugins into separate library and load dynamically

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE "Release" CACHE STRING
         "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
         FORCE )
endif( NOT CMAKE_BUILD_TYPE )

add_definitions( -DLLOG_MAX_LEVEL=legion::Log::DEBUG )

if(WIN32)
  # We should set this macro as well to get our nice trig functions
  add_definitions(-D_USE_MATH_DEFINES)
  # Microsoft does some stupid things like #define min and max.
  add_definitions(-DNOMINMAX)
endif()


if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" )
  #set(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden -fvisibility=hidden -Wall -Wextra -pedantic  -funroll-loops -floop-optimize -ffast-math -Qunused-arguments" )
  set(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden -fvisibility=hidden -Wall -Wextra -Werror -funroll-loops -floop-optimize -ffast-math -Qunused-arguments" )
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fvisibility-inlines-hidden -fvisibility=hidden -Wall -Wextra -Werror -funroll-loops -floop-optimize -ffast-math" )
endif()


if(NOT WIN32)
  set(CMAKE_CXX_FLAGS_DEBUG          "-g")
  set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
endif()

find_package( OptiX    REQUIRED )
find_package( CUDA 4.0 REQUIRED )

include_directories(
    "${CMAKE_BINARY_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CUDA_INCLUDE_DIRS}" 
    "${OptiX_INCLUDE}"
    )

# Tell clang_complete about our include paths
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/clang_complete.in"  "${CMAKE_CURRENT_SOURCE_DIR}/.clang_complete" )


add_subdirectory( Support )
add_subdirectory( Legion)
add_subdirectory( Standalone )
add_subdirectory( Test )
