
#ifndef LEGION_INTERFACE_IFILM_H_
#define LEGION_INTERFACE_IFILM_H_

#include <private/APIBase.hpp>
#include <Core/Index.hpp>
#include <Core/Color.hpp>

namespace legion
{


class IFilm : public APIBase
{
public:
    IFilm( const std::string& name );
    virtual ~IFilm();

    virtual void addSample( const Index2& pixel_index, const Color& color, float weight )=0;
    virtual void shutterOpen()=0;
    virtual void shutterClose()=0;
    virtual void passComplete()=0;
    virtual void setDimensions( const Index2& dimensions )=0;
    virtual Index2  getDimensions()=0;
};


} // namespace legion

#endif // LEGION_INTERFACE_IFILM_H_
