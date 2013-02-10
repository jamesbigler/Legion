
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/src" )

add_executable( lr 
    src/lr.cpp
    src/XMLToLegion.cpp
    src/XMLToLegion.hpp
    )
target_link_libraries( lr legion )

