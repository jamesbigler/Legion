
#include <Legion/Core/Exception.hpp>
#include <string>

using namespace legion;


Exception::Exception()
    : std::runtime_error( "Legion exception" )
{
}


Exception::Exception( const std::string& mssg )
    : std::runtime_error( mssg )
{
}


AssertionFailure::AssertionFailure()
    : Exception( "Legion assertion failure" )
{
}


AssertionFailure::AssertionFailure( const std::string& mssg )
    : Exception( "Legion assertion failure: " + mssg )
{
}
