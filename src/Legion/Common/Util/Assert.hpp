

#ifndef LEGION_COMMON_UTIL_ASSERT_H_
#define LEGION_COMMON_UTIL_ASSERT_H_

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

/// Static (compile time) assertion.
/// \param condition   The condition to be tested
#define LEGION_STATIC_ASSERT( condition )                                      \
    typedef StaticAssertionChecker<                                            \
    sizeof( StaticAssertionFailure<(bool)(condition)> ) >                      \
    LEGION_JOIN( _static_assertion_checker_, __LINE__ )

}

#endif // LEGION_COMMON_UTIL_ASSERT_H_
