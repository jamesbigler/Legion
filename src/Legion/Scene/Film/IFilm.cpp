
#include <Legion/Scene/Film/IFilm.hpp>

using namespace legion;


IFilm::IFilm( Context* context, const std::string& name )
    : APIBase( context, name )
{
}


IFilm::~IFilm()
{
}
