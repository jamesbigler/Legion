
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License) Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following
// conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <Legion/Common/Math/Math.hpp>
#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Image.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Scene/Film/ImageFilm.hpp>

#include <iomanip>
#include <cstring> // memset

using namespace legion;

//------------------------------------------------------------------------------
//
// ImageFilm::Impl
//
//------------------------------------------------------------------------------



ImageFilm::ImageFilm()
{
}


void ImageFilm::setVariables( VariableContainer& container ) const
{
}

/*
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
    unsigned idx = getIndex( pixel_index, m_dimensions );
    const Color orig_color  = m_data[ idx ];
    const float orig_weight = m_weights[ idx ];
    const float new_weight  = orig_weight + weight;
    const Color new_color   = lerp( orig_color, color, weight/new_weight );

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
}


void ImageFilm::shutterClose()
{
    writeOpenEXR( "output.exr", m_dimensions.x(), m_dimensions.y(), 3,
                  reinterpret_cast<float*>( m_data ) );
}


void ImageFilm::passComplete()
{
}
*/



