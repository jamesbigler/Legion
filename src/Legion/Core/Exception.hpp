
#ifndef LEGION_CORE_EXCEPTION_HPP_
#define LEGION_CORE_EXCEPTION_HPP_

#include <stdexcept>

namespace legion
{

class Exception : public std::runtime_error
{
public:
    Exception();
    Exception( const std::string& mssg );
};


}

#endif // LEGION_CORE_EXCEPTION_HPP_
