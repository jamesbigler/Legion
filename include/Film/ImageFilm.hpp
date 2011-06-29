
#ifndef LEGION_IMAGE_FILM_H_
#define LEGION_IMAGE_FILM_H_

#include <Interface/IFilm.hpp>

namespace legion
{


class ImageFilm : public IFilm 
{
public:
    ImageFilm( const std::string& name );
    ~ImageFilm();
    
    void   setDimensions( const Index2& dimensions );
    Index2 getDimensions()const;

    void addSample( const Index2& pixel_index, const Color& color, float weight );
    void shutterOpen();
    void shutterClose();
    void passComplete();

private:
    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};

} // namespace legion

#endif // LEGION_IMAGE_FILM_H_
