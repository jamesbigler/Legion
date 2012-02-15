
#include <Scene/Film/ImageFilm.hpp>
#include <Util/Stream.hpp>
#include <iostream>

using namespace legion;

//------------------------------------------------------------------------------
//
// ImageFilm::Impl
//
//------------------------------------------------------------------------------

class ImageFilm::Impl
{
public:
    Impl();
    ~Impl();
    
    void   setDimensions( const Index2& dimensions );
    Index2 getDimensions()const;

    void addSample( const Index2& pixel_index, const Color& color, float weight );
    void shutterOpen();
    void shutterClose();
    void passComplete();

private:
    Index2 m_dimensions;

};


ImageFilm::Impl::Impl()
    : m_dimensions( 0u, 0u )
{
}

ImageFilm::Impl::~Impl()
{
}

void ImageFilm::Impl::setDimensions( const Index2& dimensions )
{
    m_dimensions = dimensions;
}


Index2 ImageFilm::Impl::getDimensions()const
{
    return m_dimensions; 
}


void ImageFilm::Impl::addSample( const Index2& pixel_index, const Color& color, float weight )
{
    std::cerr << "ImageFilm::addSample( " << pixel_index << ", " << color << ", " << weight << " )" << std::endl;
}


void ImageFilm::Impl::shutterOpen()
{
    std::cerr << "ImageFilm::shutterOpen()" << std::endl;
}


void ImageFilm::Impl::shutterClose()
{
    std::cerr << "ImageFilm::shutterClose()" << std::endl;
}


void ImageFilm::Impl::passComplete()
{
    std::cerr << "ImageFilm::passComplete()" << std::endl;
}



//------------------------------------------------------------------------------
//
// ImageFilm
//
//------------------------------------------------------------------------------

ImageFilm::ImageFilm( const std::string& name )
    : IFilm( name ),
      m_impl( new Impl )
{
}


ImageFilm::~ImageFilm()
{
}


void ImageFilm::setDimensions( const Index2& dimensions )
{
    m_impl->setDimensions( dimensions );
}


Index2 ImageFilm::getDimensions()const
{
    return m_impl->getDimensions(); 
}


void ImageFilm::addSample( const Index2& pixel_index, const Color& color, float weight )
{
    m_impl->addSample( pixel_index, color, weight );
}


Color ImageFilm::getPixel( const Index2& pixel_index )const
{
    return Color();
}


Color* ImageFilm::getPixels()const
{
    return 0u;
}


void ImageFilm::shutterOpen()
{
    m_impl->shutterOpen();
}


void ImageFilm::shutterClose()
{
    m_impl->shutterClose();
}


void ImageFilm::passComplete()
{
    m_impl->passComplete();
}


