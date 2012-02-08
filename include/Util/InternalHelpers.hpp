

#ifndef LEGION_UTIL_INTERNAL_HELPERS_HPP_
#define LEGION_UTIL_INTERNAL_HELPERS_HPP_

/// The global namespace for the Legion API
namespace legion 
{

//------------------------------------------------------------------------------
//
// Compile time  assertion helpers.
//
//------------------------------------------------------------------------------

template<bool>   struct StaticAssertionFailure;
template<>       struct StaticAssertionFailure<true> {};
template<size_t> struct StaticAssertionChecker       {};

#define LEGION_JOIN( X, Y ) LEGION_DO_JOIN( X, Y )
#define LEGION_DO_JOIN( X, Y ) LEGION_DO_JOIN2(X,Y)
#define LEGION_DO_JOIN2( X, Y ) X##Y

}

#endif // LEGION_UTIL_INTERNAL_HELPERS_HPP_
