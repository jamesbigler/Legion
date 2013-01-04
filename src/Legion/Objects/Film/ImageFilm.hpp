
#ifndef LEGION_OBJECTS_FILM_IMAGEFILM_H_
#define LEGION_OBJECTS_FILM_IMAGEFILM_H_

#include <Legion/Objects/Film/IFilm.hpp>

namespace legion
{


class ImageFilm : public IFilm
{
public:
    ImageFilm();
    
    void setVariables( VariableContainer& container ) const;
    /*
    ImageFilm( Context* context, const std::string& name );
    ~ImageFilm();
    
    void   setDimensions( const Index2& dimensions );
    Index2 getDimensions()const;

    void  addSample( const Index2& pixel_index,
                     float weight,
                     const Color& color );

    Color getPixel( const Index2& pixel_index )const;
    Color* getPixels()const;

    void shutterOpen();
    void shutterClose();
    void passComplete();
    */

private:

    /*
    Index2     m_dimensions;
    Color*     m_data;
    float*     m_weights;
    */


};

} // namespace legion

#endif // LEGION_OBJECTS_FILM_IMAGEFILM_H_
