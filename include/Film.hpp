
#ifndef LEGION_FILM_H_
#define LEGION_FILM_H_


namespace legion
{


class IFilm
{
public:
    IFilm( const std::string& name );

    virtual ~IFilm();

    virtual void addSample( const Index2& pixel_index, const Color& color, float weight )=0;
    virtual void shutterOpen=0;
    virtual void shutterClose=0;
    virtual void passComplete=0; 
};


} // namespace legion

#endif // LEGION_FILM_H_
