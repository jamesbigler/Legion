

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/src" )

###############################################################################
#
#  Comand line lr renderer
#
###############################################################################

set( lr_headers
    src/Util.hpp
    src/XMLToLegion.hpp
    )

set( lr_sources
    src/Util.cpp
    src/XMLToLegion.cpp
    )

add_executable( lr 
    src/lr.cpp
    ${lr_sources}
    ${lr_headers}
    )

target_link_libraries( lr
    legion 
    )


###############################################################################
#
#  Qt gui lr renderer
#
###############################################################################

find_package( Qt4 )
if( QT4_FOUND AND FALSE)
    include( ${QT_USE_FILE} )
    add_definitions( ${QT_DEFINITIONS} )
    include_directories( ${QT_INCLUDES} )

    # TODO: implement ImageFileDisplay fallback when Qt not found

    set( lrgui_sources
        src/gui/DisplayWidget.cpp
        src/gui/ImageWidget.cpp
        src/gui/LegionDisplay.cpp
        src/gui/RenderThread.cpp
        src/gui/Window.cpp
        )

    set( lrgui_headers
        src/gui/DisplayWidget.hpp
        src/gui/ImageWidget.hpp
        src/gui/LegionDisplay.hpp
        src/gui/RenderThread.hpp
        src/gui/Window.hpp
        )

    QT4_WRAP_CPP( lrgui_headers_moc ${lrgui_headers})

    add_executable( lrgui
        src/lrgui.cpp
        ${lr_sources}
        ${lr_headers}
        ${lrgui_sources}
        ${lrgui_headers_moc}
        )

    target_link_libraries( lrgui
        legion 
        ${QT_LIBRARIES}
        )

endif()
