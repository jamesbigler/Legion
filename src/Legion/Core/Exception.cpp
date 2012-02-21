
#include <Legion/Core/Exception.hpp>

using namespace legion;


Exception::Exception()
    : std::runtime_error( "Legion exception" )
{
}


Exception::Exception( const std::string& mssg )
    : std::runtime_error( mssg )
{
}
