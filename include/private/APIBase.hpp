
#ifndef LEGION_APIBASE_H_
#define LEGION_APIBASE_H_

#include <string>
#include <tr1/memory>

namespace legion
{


class APIBase
{
public:
    APIBase( const std::string& name );
    virtual ~APIBase();

    std::string getName()const;

private:
    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};


}

#endif // LEGION_APIBASE_H_
