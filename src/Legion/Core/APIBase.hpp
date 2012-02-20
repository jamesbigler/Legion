
#ifndef LEGION_CORE_APIBASE_H_
#define LEGION_CORE_APIBASE_H_

#include <string>
#include <Legion/Common/Util/Noncopyable.hpp>

namespace legion
{

class Context;

class APIBase : public Noncopyable
{
public:
    APIBase( Context* context, const std::string& name );
    virtual ~APIBase();

    std::string    getName()const;

    unsigned       getID()const;

    Context&       getContext();
    const Context& getContext()const;

private:
    const std::string   m_name;
    const unsigned      m_uid;
    Context*            m_context;

    static unsigned     s_next_uid;
};

}

#endif // LEGION_CORE_APIBASE_H_
