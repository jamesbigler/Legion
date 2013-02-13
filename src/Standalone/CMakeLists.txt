
add_executable( simple simple.cpp)
target_link_libraries( simple legion )

add_executable( ward ward.cpp)
target_link_libraries( ward legion )

add_executable( teapot teapot.cpp)
target_link_libraries( teapot legion )

add_executable( texture texture.cpp)
target_link_libraries( texture legion )

add_subdirectory( lr )
add_subdirectory( lr2 )

#add_executable( sphere sphere.cpp )
#target_link_libraries( sphere legion)


#add_executable( spheres spheres.cpp )
#target_link_libraries( spheres legion)

#add_executable( cornell cornell.cpp )
#target_link_libraries( cornell legion)
