
#ifndef LEGION_CORE_APIBASE_H_
#define LEGION_CORE_APIBASE_H_

#include <string>
#include <Legion/Common/Util/Noncopyable.hpp>

namespace legion
{

class APIBase : public Noncopyable
{
public:
    APIBase( const std::string& name );
    virtual ~APIBase();

    std::string getName()const;

    unsigned    getID()const;

private:
    const std::string   m_name;
    const unsigned      m_uid;

    static unsigned     s_next_uid;
};

}

#endif // LEGION_CORE_APIBASE_H_
