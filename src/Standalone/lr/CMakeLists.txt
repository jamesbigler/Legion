

find_package( Qt4 REQUIRED QtCore QtGui )
include( ${QT_USE_FILE} )
ADD_DEFINITIONS(${QT_DEFINITIONS})

include_directories( 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${QT_INCLUDES}
    )
# TODO: implement ImageFileDisplay fallback when Qt not found

set( lr_headers
    src/Util.hpp
    src/XMLToLegion.hpp
    src/gui/GUI.hpp
    src/gui/QtDisplay.hpp
    src/gui/Window.hpp
    )

set( lr_sources
    src/lr.cpp
    src/Util.cpp
    src/XMLToLegion.cpp
    src/gui/GUI.cpp
    src/gui/QtDisplay.cpp
    )

QT4_WRAP_CPP( lr_headers_moc ${lr_headers})

add_executable( lr 
    ${lr_sources}
    ${lr_headers}
    )

target_link_libraries( lr
    legion 
    ${QT_LIBRARIES}
    )

