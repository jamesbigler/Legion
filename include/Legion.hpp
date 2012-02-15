
#ifndef LEGION_LEGION_HPP_
#define LEGION_LEGION_HPP_

/// \file Legion.hpp
/// Wrapper include for the Legion API.  Includes all of the public Legion
/// headers.

#include <Core/APIBase.hpp>
#include <Core/Color.hpp>
#include <Core/Context.hpp>
#include <Core/Exception.hpp>
#include <Core/Matrix.hpp>
#include <Core/Mesh.hpp>
#include <Core/Ray.hpp>
#include <Core/Vector.hpp>
#include <Interface/ICamera.hpp>
#include <Interface/IFilm.hpp>
#include <Interface/ILightShader.hpp>
#include <Interface/ISurfaceShader.hpp>
#include <Legion.hpp>
#include <Util/InternalHelpers.hpp>
#include <Util/Math.hpp>
#include <Util/Stream.hpp>
#include <Util/Util.hpp>

// These will go away when factories are in place
#include <../src/Scene/Camera/ThinLensCamera.hpp>
#include <../src/Scene/Film/ImageFilm.hpp>
#include <../src/Scene/LightShader/PointLightshader.hpp>
#include <../src/Scene/SurfaceShader/LambertianShader.hpp>


#endif // LEGION_LEGION_HPP_
