
add_library( legion SHARED
   ../include/Camera/ThinLensCamera.hpp
   ../include/Core/Color.hpp
   ../include/Core/Context.hpp
   ../include/Core/Exception.hpp
   ../include/Core/Index.hpp
   ../include/Core/Matrix.hpp
   ../include/Core/Mesh.hpp
   ../include/Core/Ray.hpp
   ../include/Core/Vector.hpp
   ../include/Film/Film.hpp
   ../include/Film/ImageFilm.hpp
   ../include/Interface/IBasicCamera.hpp
   ../include/Interface/ICamera.hpp
   ../include/Interface/IFilm.hpp
   ../include/Interface/ILightShader.hpp
   ../include/Interface/ISurfaceShader.hpp
   ../include/LightShader/PointLightShader.hpp
   ../include/Private/APIBase.hpp
   ../include/SurfaceShader/LambertianShader.hpp
   ../include/Util/Stream.hpp
   ../include/Util/Util.h
   Camera/IBasicCamera.cpp
   Camera/ICamera.cpp
   Camera/ThinLensCamera.cpp
   Core/APIBase.cpp
   Core/Context.cpp
   Engine/Engine.cpp
   Film/IFilm.cpp
   Film/ImageFilm.cpp
   LightShader/ILightShader.cpp
   LightShader/PointLightShader.cpp
   Math/Math.cpp
   Mesh/Mesh.cpp
   SurfaceShader/ISurfaceShader.cpp
   SurfaceShader/LambertianShader.cpp
   Util/Stream.cpp
   )

#TODO: Move plugins into separate library and load dynamically
