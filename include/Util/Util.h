
#ifndef LEGION_UTIL_UTIL_HPP_
#define LEGION_UTIL_UTIL_HPP_


//------------------------------------------------------------------------------
//
// Compile time  assert.
//
//------------------------------------------------------------------------------

namespace legion 
{

template<bool> struct StaticAssertionFailure;
template<> struct StaticAssertionFailure<true> {};
template<size_t> struct StaticAssertionChecker {};

#define LEGION_JOIN( X, Y ) LEGION_DO_JOIN( X, Y )
#define LEGION_DO_JOIN( X, Y ) LEGION_DO_JOIN2(X,Y)
#define LEGION_DO_JOIN2( X, Y ) X##Y
#define LEGION_STATIC_ASSERT( condition )                                      \
    typedef StaticAssertionChecker<                                            \
    sizeof( StaticAssertionFailure<(bool)(condition)> ) >                      \
    LEGION_JOIN( _static_assertion_checker_, __LINE__ )


}

#endif // LEGION_UTIL_UTIL_HPP_
