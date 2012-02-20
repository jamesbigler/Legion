
#ifndef LEGION_COMMON_UTIL_SINGLETON_HPP_
#define LEGION_COMMON_UTIL_SINGLETON_HPP_

#include <Legion/Common/Util/Noncopyable.hpp>

namespace legion
{

/// Requires T to have a default constructor.  If non-default constructor is
/// needed a factory function template parameter can be added.
template <typename T>
class Singleton : Noncopyable
{
public:
    static const T& instance()  { static T inst; return inst; } 

protected:
    Singleton() {}

};

}


#endif // LEGION_COMMON_UTIL_SINGLETON_HPP_
