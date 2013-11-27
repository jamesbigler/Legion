
#ifndef LEGION_CORE_EXCEPTION_HPP_
#define LEGION_CORE_EXCEPTION_HPP_

#include <stdexcept>
#include <Legion/Common/Util/Preprocessor.hpp>

namespace legion
{

class Exception : public std::runtime_error
{
public:
    LAPI Exception();
    LAPI explicit Exception( const std::string& mssg );
};


class AssertionFailure : public Exception
{
public:
    LAPI AssertionFailure();
    LAPI explicit AssertionFailure( const std::string& mssg );
};

}

#endif // LEGION_CORE_EXCEPTION_HPP_
