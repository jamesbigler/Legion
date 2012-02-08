
/// \file Util.hpp
/// Utility functions and macros

#ifndef LEGION_UTIL_UTIL_HPP_
#define LEGION_UTIL_UTIL_HPP_

#include <Util/InternalHelpers.h>

namespace legion 
{

/// Static (compile time) assertion.
/// \param condition   The condition to be tested
#define LEGION_STATIC_ASSERT( condition )                                      \
    typedef StaticAssertionChecker<                                            \
    sizeof( StaticAssertionFailure<(bool)(condition)> ) >                      \
    LEGION_JOIN( _static_assertion_checker_, __LINE__ )


/// Warp the uniformly chosen 2D sample to fit a Box filter kernel.
/// \param in_sample  The input  sample in [0,2]^2
/// \returns          The warped sample in [-0.5,0.5]^2
Vector2 warpSampleByBoxFilter        ( const Vector2& in_sample );

/// Warp the uniformly chosen 2D sample to fit a Tent filter kernel.
///   \param in_sample  The input  sample in [0,2]^2
///   \returns          The warped sample in [-1,1]^2
Vector2 warpSampleByTentFilter       ( const Vector2& in_sample );

/// Warp the uniformly chosen 2D sample to fit a Cubic Spline filter kernel.
///   \param in_sample  The input  sample in [0,2]^2
///   \returns          The warped sample in [-2,2]^2
Vector2 warpSampleByCubicSplineFilter( const Vector2& in_sample );

}

#endif // LEGION_UTIL_UTIL_HPP_
