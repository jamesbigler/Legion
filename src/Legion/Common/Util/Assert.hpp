

#ifndef LEGION_COMMON_UTIL_ASSERT_H_
#define LEGION_COMMON_UTIL_ASSERT_H_

#include <Legion/Core/Exception.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
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
    typedef legion::StaticAssertionChecker<                                    \
    sizeof( legion::StaticAssertionFailure<(bool)(condition)> ) >              \
    LEGION_JOIN( _static_assertion_checker_, __LINE__ )


//------------------------------------------------------------------------------
//
//  LEGION_TODO triggers exception. Place this in unimplemented functions, etc
//
//------------------------------------------------------------------------------
#define LEGION_TODO()                                                          \
    throw legion::Exception( std::string( LFUNC ) +              \
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
            throw legion::AssertionFailure( ss.str() );                        \
        }                                                                      \
    } while( 0 )

}

#define LEGION_ASSERT_POINTER_PARAM( p )                                       \
    do                                                                         \
    {                                                                          \
        if( !(p) )                                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << LFUNC <<  " passed NULL input pointer '" << #p << "'";    \
            throw legion::AssertionFailure( ss.str() );                        \
        }                                                                      \
    } while( 0 )
    

#endif // LEGION_COMMON_UTIL_ASSERT_H_
