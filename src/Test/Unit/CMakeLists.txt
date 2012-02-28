
add_executable( unit_vector 
  unit_vector.cpp
  )

add_executable( unit_image
  unit_image.cpp
  )
target_link_libraries( unit_image
  legion
  )
