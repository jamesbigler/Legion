
#include <Private/APIBase.hpp>
#include <string>


using namespace legion;

/******************************************************************************\
 *
 *
 *
\******************************************************************************/

namespace legion
{
    class APIBase::Impl
    {
    public:
        Impl(const std::string& name );
        
        std::string getName()const;
    private:
        std::string m_name;
    };

    APIBase::Impl::Impl( const std::string& name )
        : m_name( name )
    {
    }


    std::string APIBase::Impl::getName()const
    {
        return m_name;
    }
}

/******************************************************************************\
 *
 *
 *
\******************************************************************************/

APIBase::APIBase( const std::string& name )
    : m_impl( new Impl( name ) )
{
}

  
APIBase::~APIBase()
{
}


std::string APIBase::getName()const
{
    return m_impl->getName();
}

