
#ADD_DEFINITIONS ( -DHAVE_CONFIG_H -DILM_IMF_TEST_IMAGEDIR="${CMAKE_SOURCE_DIR}/openexr/IlmImfTest/" )


INCLUDE_DIRECTORIES (
  "${CMAKE_BINARY_DIR}/include"
  IlmImf
  exrmaketiled
  exrenvmap
  exrmakepreview
  exrmultiview
  IlmImfFuzzTest
)

# ilmbase
INCLUDE_DIRECTORIES (
  "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/Half"
  "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/Iex"
  "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/Imath"
  "${CMAKE_SOURCE_DIR}/src/Support/ilmbase-2.0.1/IlmThread"
  "${CMAKE_SOURCE_DIR}/src/Support/zlib-1.2.8"
)

IF (WIN32)
  INCLUDE_DIRECTORIES (
    ${ZLIB_ROOT}/include
    )
  LINK_DIRECTORIES (
    ${ZLIB_ROOT}/lib
    )
  SET ( Z_LIB zlib )
ELSE()
  SET ( PTHREAD_LIB pthread )
  #SET ( Z_LIB z )
  SET ( Z_LIB zlib )
ENDIF()

INCLUDE_DIRECTORIES ( ${ILMBASE_PACKAGE_PREFIX}/include/OpenEXR )
LINK_DIRECTORIES ( ${ILMBASE_PACKAGE_PREFIX}/lib )

ADD_SUBDIRECTORY ( IlmImf )
#ADD_SUBDIRECTORY ( IlmImfTest )
#ADD_SUBDIRECTORY ( IlmImfFuzzTest )
#ADD_SUBDIRECTORY ( IlmImfExamples )

#ADD_SUBDIRECTORY ( exrheader )
#ADD_SUBDIRECTORY ( exrmaketiled )
#ADD_SUBDIRECTORY ( exrstdattr )
#ADD_SUBDIRECTORY ( exrmakepreview )
#ADD_SUBDIRECTORY ( exrenvmap )
#ADD_SUBDIRECTORY ( exrmultiview )
#ADD_SUBDIRECTORY ( exr2aces )
#ADD_SUBDIRECTORY ( exrmultipart )

##########################
# Installation
##########################

# Documentation
INSTALL ( FILES
  doc/TechnicalIntroduction.pdf
  doc/TechnicalIntroduction_2.0.pdf
  doc/ReadingAndWritingImageFiles.pdf
  doc/OpenEXRFileLayout.pdf
  doc/OpenEXRFileLayout_2.0.pdf
  doc/ReadingAndWritingImageFiles_2.0.pdf
  doc/MultiViewOpenEXR.pdf
  DESTINATION
  ${CMAKE_INSTALL_PREFIX}/share/doc/OpenEXR-2.0.0
  )

# Examples

INSTALL ( FILES
  IlmImfExamples/main.cpp
  IlmImfExamples/drawImage.cpp
  IlmImfExamples/rgbaInterfaceExamples.cpp
  IlmImfExamples/rgbaInterfaceTiledExamples.cpp
  IlmImfExamples/generalInterfaceExamples.cpp
  IlmImfExamples/lowLevelIoExamples.cpp
  IlmImfExamples/previewImageExamples.cpp
  IlmImfExamples/generalInterfaceTiledExamples.cpp
  IlmImfExamples/generalInterfaceTiledExamples.h
  IlmImfExamples/drawImage.h
  IlmImfExamples/rgbaInterfaceExamples.h
  IlmImfExamples/generalInterfaceExamples.h
  IlmImfExamples/rgbaInterfaceTiledExamples.h
  IlmImfExamples/lowLevelIoExamples.h
  IlmImfExamples/previewImageExamples.h
  DESTINATION
  ${CMAKE_INSTALL_PREFIX}/share/doc/OpenEXR-2.0.0/examples
  )
