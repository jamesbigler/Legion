
#include <Film/ImageFilm.hpp>

using namespace legion;

ImageFilm::ImageFilm( const std::string& name )
  : IFilm( name )
{
}


ImageFilm::~ImageFilm()
{
}


void ImageFilm::addSample( const Index2& pixel_index, const Color& color, float weight )
{
}


void ImageFilm::shutterOpen()
{
}


void ImageFilm::shutterClose()
{
}


void ImageFilm::passComplete()
{
}

void ImageFilm::setDimensions( const Index2& dimensions )
{
}


Index2 ImageFilm::getDimensions()
{
}
