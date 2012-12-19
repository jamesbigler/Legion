

#ifndef LEGION_COMMON_UTIL_ASSERT_H_
#define LEGION_COMMON_UTIL_ASSERT_H_

#include <Legion/Core/Exception.hpp>
#include <string>
#include <sstream>

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

#define LEGION_JOIN( X, Y )     LEGION_DO_JOIN( X, Y )
#define LEGION_DO_JOIN( X, Y )  LEGION_DO_JOIN2(X,Y)
#define LEGION_DO_JOIN2( X, Y ) X##Y

/// Static (compile time) assertion.
/// \param condition   The condition to be tested
#define LEGION_STATIC_ASSERT( condition )                                      \
    typedef StaticAssertionChecker<                                            \
    sizeof( StaticAssertionFailure<(bool)(condition)> ) >                      \
    LEGION_JOIN( _static_assertion_checker_, __LINE__ )


//------------------------------------------------------------------------------
//
//  LEGION_TODO triggers exception. Place this in unimplemented functions, etc
//
//------------------------------------------------------------------------------
#define LEGION_TODO()                                                          \
    throw Exception( std::string( __PRETTY_FUNCTION__ ) +                      \
                     ": Unimplemented code path taken (TODO)");


//------------------------------------------------------------------------------
//
//  LEGION_ASSERT. triggers exception.
//
//------------------------------------------------------------------------------
#define LEGION_ASSERT( cond )                                                  \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << __FILE__ << " (" << __LINE__ << "): " << #cond;              \
            throw AssertionFailure( ss.str() );                                \
        }                                                                      \
    } while( 0 )

}

#endif // LEGION_COMMON_UTIL_ASSERT_H_
