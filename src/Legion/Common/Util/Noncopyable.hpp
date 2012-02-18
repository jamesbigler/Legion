
#ifndef LEGION_COMMON_UTIL_NONCOPYABLE_HPP_
#define LEGION_COMMON_UTIL_NONCOPYABLE_HPP_

namespace legion
{

class Noncopyable
{
protected:
    Noncopyable()  {}
    ~Noncopyable() {}

private:
    Noncopyable( const Noncopyable& );
    const Noncopyable& operator=( const Noncopyable& );
};

}


#endif //LEGION_COMMON_UTIL_NONCOPYABLE_HPP_
