
#ifndef LEGION_APIBASE_H_
#define LEGION_APIBASE_H_

#include <memory>

class APIBase
{
public:
    APIBase( const std::string& name );
    virtual ~APIBase();

    const std::string getName();

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

#endif // LEGION_APIBASE_H_
