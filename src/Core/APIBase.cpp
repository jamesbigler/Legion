
#include <Private/APIBase.hpp>
#include <string>


using namespace legion;


/******************************************************************************\
 *
 *
 *
\******************************************************************************/

APIBase::APIBase( const std::string& name )
    : m_name(  name  )
{
}

  
APIBase::~APIBase()
{
}


std::string APIBase::getName()const
{
    return m_name;
}

