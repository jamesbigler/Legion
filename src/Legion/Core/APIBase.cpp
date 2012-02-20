
#include <Legion/Core/APIBase.hpp>
#include <string>


using namespace legion;


/******************************************************************************\
 *
 *
 *
\******************************************************************************/

unsigned APIBase::s_next_uid = 0u;


APIBase::APIBase( Context* context, const std::string& name )
    : m_context( context ),
      m_name( name ),
      m_uid( s_next_uid++ )
{
}

  
APIBase::~APIBase()
{
}


std::string APIBase::getName()const
{
    return m_name;
}


unsigned APIBase::getID()const
{
    return m_uid;
}


Context& APIBase::getContext()
{
    return *m_context;
}


const Context& APIBase::getContext()const
{
    return *m_context;
}
