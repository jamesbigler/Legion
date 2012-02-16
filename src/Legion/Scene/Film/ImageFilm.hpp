
#ifndef LEGION_SCENE_FILM_IMAGEFILM_H_
#define LEGION_SCENE_FILM_IMAGEFILM_H_

#include <Legion/Scene/Film/IFilm.hpp>
#include <tr1/memory>

namespace legion
{


class ImageFilm : public IFilm 
{
public:
    ImageFilm( const std::string& name );
    ~ImageFilm();
    
    void   setDimensions( const Index2& dimensions );
    Index2 getDimensions()const;

    void  addSample( const Index2& pixel_index, const Color& color, float weight );

    Color getPixel( const Index2& pixel_index )const;
    Color* getPixels()const;

    void shutterOpen();
    void shutterClose();
    void passComplete();

private:
    class Impl;
    std::tr1::shared_ptr<Impl> m_impl;
};

} // namespace legion

#endif // LEGION_SCENE_FILM_IMAGEFILM_H_
