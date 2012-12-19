
#ifndef LEGION_CORE_EXCEPTION_HPP_
#define LEGION_CORE_EXCEPTION_HPP_

#include <stdexcept>

namespace legion
{

class Exception : public std::runtime_error
{
public:
    Exception();
    explicit Exception( const std::string& mssg );
};


class AssertionFailure : public Exception
{
public:
    AssertionFailure();
    explicit AssertionFailure( const std::string& mssg );
};

}

#endif // LEGION_CORE_EXCEPTION_HPP_
