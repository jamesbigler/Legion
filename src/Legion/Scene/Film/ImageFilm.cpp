
#include <Legion/Scene/Film/ImageFilm.hpp>
#include <Legion/Core/Vector.hpp>
#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Common/Util/Image.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <iomanip>
#include <cstring> // memset

using namespace legion;

//------------------------------------------------------------------------------
//
// ImageFilm::Impl
//
//------------------------------------------------------------------------------



namespace
{
    inline unsigned getIndex( const Index2& index2, const Index2& dim  )
    {
        return index2.y() * dim.x() + index2.x();
    }
}


ImageFilm::ImageFilm( Context* context, const std::string& name )
    : IFilm( context, name ),
      m_dimensions( 0u, 0u ),
      m_data( 0u )
{
}


ImageFilm::~ImageFilm()
{
    delete [] m_data;
    delete [] m_weights;
}


void ImageFilm::setDimensions( const Index2& dimensions )
{
    m_dimensions = dimensions;
    if( m_data )
    {
        delete [] m_data;
        delete [] m_weights;
    }
    m_data    = new Color[ dimensions.x() * dimensions.y() ];
    m_weights = new float[ dimensions.x() * dimensions.y() ];

    memset( m_data,    0, dimensions.x() * dimensions.y() * sizeof( Color ) );
    memset( m_weights, 0, dimensions.x() * dimensions.y() * sizeof( float ) );
}


Index2 ImageFilm::getDimensions()const
{
    return m_dimensions; 
}


void ImageFilm::addSample( const Index2& pixel_index,
                           float weight,
                           const Color& color )
{
    // TODO: use weights to combine
    unsigned idx = getIndex( pixel_index, m_dimensions );
    const Color orig_color  = m_data[ idx ];
    const float orig_weight = m_weights[ idx ];
    const float new_weight  = orig_weight + weight;
    const Color new_color   = lerp( orig_color, color, weight/new_weight );

    

    /*
    LLOG_INFO << std::setprecision(3) 
              << color << ":" << orig_color << "->" << new_color << "  " 
              << orig_weight << ":" << new_weight;
              */

    m_data[ idx ] = new_color;
    m_weights[ idx ] = new_weight;
}


Color ImageFilm::getPixel( const Index2& pixel_index )const
{
    return m_data[ getIndex( pixel_index, m_dimensions ) ];
}


Color* ImageFilm::getPixels()const
{
    return m_data;
}


void ImageFilm::shutterOpen()
{
    LLOG_INFO << "ImageFilm::shutterOpen()";
}


void ImageFilm::shutterClose()
{
    LLOG_INFO << "ImageFilm::shutterClose()";
    writeOpenEXR( "output.exr", m_dimensions.x(), m_dimensions.y(), 3,
                  reinterpret_cast<float*>( m_data ) );
}


void ImageFilm::passComplete()
{
    LLOG_INFO << "ImageFilm::passComplete()";
}



