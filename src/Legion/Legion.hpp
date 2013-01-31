
#ifndef LEGION_LEGION_HPP_
#define LEGION_LEGION_HPP_

/// \file Legion.hpp


#include <Legion/Common/Math/Filter.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Math/Matrix.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Assert.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Core/Color.hpp>
#include <Legion/Core/Context.hpp>
#include <Legion/Core/Exception.hpp>
#include <Legion/Objects/Renderer/IRenderer.hpp>
#include <Legion/Objects/Renderer/ProgressiveRenderer.hpp>
#include <Legion/Objects/Camera/ICamera.hpp>
#include <Legion/Objects/Camera/ThinLens.hpp>
#include <Legion/Objects/Display/IDisplay.hpp>
#include <Legion/Objects/Display/ImageFileDisplay.hpp>
#include <Legion/Objects/Environment/IEnvironment.hpp>
#include <Legion/Objects/Environment/ConstantEnvironment.hpp>
#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Objects/Surface/Lambertian.hpp>
#include <Legion/Objects/Surface/Ward.hpp>
#include <Legion/Objects/Surface/DiffuseEmitter.hpp>
#include <Legion/Objects/Geometry/IGeometry.hpp>
#include <Legion/Objects/Geometry/Parallelogram.hpp>
#include <Legion/Objects/Geometry/Sphere.hpp>
#include <Legion/Objects/Geometry/TriMesh.hpp>
#include <Legion/Objects/Light/ILight.hpp>
#include <Legion/Objects/Light/PointLight.hpp>

#endif // LEGION_LEGION_HPP_
