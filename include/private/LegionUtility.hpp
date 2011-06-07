
#ifndef LEGION_NONCOPYABLE_H_
#define LEGION_NONCOPYABLE_H_

class NonCopyable
{
protected:
    NonCopyable()  {}
    ~NonCopyable() {}
       
private:
    NonCopyable( const NonCopyable& );
    const NonCopyable& operator=( const NonCopyable& );
};

#endif // LEGION_NONCOPYABLE_H_
