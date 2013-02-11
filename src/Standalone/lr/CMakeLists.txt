

find_package( Qt4 REQUIRED QtCore QtGui )
include( ${QT_USE_FILE} )


include_directories( 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${QT_INCLUDES}
    )
# TODO: implement ImageFileDisplay fallback when Qt not found

add_executable( lr 
    src/lr.cpp
    src/XMLToLegion.cpp
    src/XMLToLegion.hpp
    src/gui/GUI.cpp
    src/gui/GUI.hpp
    src/gui/QtDisplay.cpp
    src/gui/QtDisplay.hpp
    src/Util.cpp
    src/Util.hpp
    )

target_link_libraries( lr
    legion 
    ${QT_LIBRARIES}
    )

