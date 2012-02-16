
#ifndef LEGION_CORE_APIBASE_H_
#define LEGION_CORE_APIBASE_H_

#include <string>

namespace legion
{


class APIBase
{
public:
    APIBase( const std::string& name );
    virtual ~APIBase();

    std::string getName()const;

private:
    std::string m_name;
};


}

#endif // LEGION_CORE_APIBASE_H_
