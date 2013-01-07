
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
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
#include <Legion/Common/Math/Sobol.hpp>
#include <Legion/Common/Util/Assert.hpp>
#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Stream.hpp>
#include <Legion/Common/Util/Util.hpp>
#include <Legion/Core/Ray.hpp>
#include <Legion/Renderer/sobol_matrices.hpp>
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Objects/Camera/ICamera.hpp>
#include <Legion/Objects/Film/IFilm.hpp>

using namespace legion;

namespace
{
    // This function courtesy of Alex Keller, based on 'Enumerating Quasi-Monte
    // Carlo Point Sequences in Elementary Intervals', Gruenschloß, Raab, and
    // Keller
    inline unsigned lookUpSobolIndex( const unsigned m,
                                      const unsigned pass,
                                      const Index2&  pixel )
    {
        typedef unsigned int       uint32;
        typedef unsigned long long uint64;
        
        const uint32 m2 = m << 1;
        uint32 frame = pass;
        uint64 index = uint64(frame) << m2;

        // TODO: the delta value only depends on frame
        // and m, thus it can be cached across multiple
        // function calls, if desired.
        uint64 delta = 0;
        for (uint32 c = 0; frame; frame >>= 1, ++c)
            if (frame & 1) // Add flipped column m + c + 1.
                delta ^= vdc_sobol_matrices[m - 1][c];

        uint64 b = ((uint64(pixel.x()) << m) | pixel.y()) ^ delta; // flipped b

        for (uint32 c = 0; b; b >>= 1, ++c)
            if (b & 1) // Add column 2 * m - c.
                index ^= vdc_sobol_matrices_inv[m - 1][c];

        return index;
    }

     
    
    void getRasterPos( const unsigned m, // 2^m should be > film max_dim
                       const unsigned pass,
                       const Index2&  pixel,
                       Vector2&       raster_pos,
                       unsigned&      sobol_index )
    {
        sobol_index = lookUpSobolIndex( m, pass, pixel );
        raster_pos.setX( static_cast<float>(Sobol::genu( sobol_index, 0 ) ) / (1U<<(32-m)) );
        raster_pos.setY( static_cast<float>(Sobol::genu( sobol_index, 1 ) ) / (1U<<(32-m)) );
    }
}


RayScheduler::RayScheduler() 
    : m_spp( 1u, 1u ),
      m_current_sample( 0u, 0u ),
      m_time_interval( 0.0f, 0.0f )
{
}


bool RayScheduler::finished()const
{
    return m_current_sample.x() >= m_spp.x() || 
           m_current_sample.y() >= m_spp.y();
}


void RayScheduler::setTimeInterval( const Vector2& time_interval )
{
    m_time_interval = time_interval;
}


void RayScheduler::setFilm( IFilm* film )
{
    m_film = film;
}


void RayScheduler::setCamera( ICamera* camera )
{
    m_camera = camera;
}


void RayScheduler::setSamplesPerPixel( const Index2& spp )
{
    m_spp = spp;
}


void RayScheduler::getPass( std::vector<Ray>& rays,
                            std::vector<PixelID>& pixel_ids )
{
    LLOG_INFO << "RayScheduler::getPass: sample " << m_current_sample;

    Index2  film_dims( m_film->getDimensions() );
    Vector2 float_film_dims( film_dims.x(), film_dims.y() );

    unsigned max_dim = film_dims.max();
    unsigned m = 0;
    while( (1u << m) < max_dim ) ++m;

    unsigned rays_per_pass = film_dims.x() * film_dims.y();
    rays.resize( rays_per_pass );
    pixel_ids.resize( rays_per_pass );

    const unsigned sample_index = index2DTo1D( m_current_sample, m_spp ); 

    unsigned ray_index = 0u;
    for( unsigned i = 0; i < film_dims.x(); ++i )
    {
        for( unsigned j = 0; j < film_dims.y(); ++j )
        {
            
            Vector2 screen_coord;
            unsigned sobol_index;
            getRasterPos( m, sample_index, Index2( i, j ), screen_coord, sobol_index );
            screen_coord /= float_film_dims;

            CameraSample sample;
            sample.screen = screen_coord; 
            sample.lens   = Vector2( Sobol::gen( sobol_index, Sobol::DIM_LENS_X ),
                                     Sobol::gen( sobol_index, Sobol::DIM_LENS_Y ) ); 
            sample.time   = lerp( m_time_interval.x(),
                                  m_time_interval.y(),
                                  Sobol::gen( sobol_index, Sobol::DIM_TIME ) ); 

            m_camera->generateRay( sample, rays[ ray_index ] );
            pixel_ids[ ray_index ].weight      = 1.0f;
            pixel_ids[ ray_index ].pixel       = Index2( i, j );
            pixel_ids[ ray_index ].sobol_index = sobol_index;;

            //LLOG_INFO << "Ray: " << rays[ ray_index ];
            ray_index++;
        }
    }

    m_current_sample = index1DTo2D( sample_index+1, m_spp );
}

