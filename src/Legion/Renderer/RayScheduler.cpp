
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
#include <Legion/Renderer/RayScheduler.hpp>
#include <Legion/Renderer/RayTracer.hpp>
#include <Legion/Scene/Camera/ICamera.hpp>
#include <Legion/Scene/Film/IFilm.hpp>

using namespace legion;

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
    Index2 film_dims       = m_film->getDimensions();
    unsigned rays_per_pass = film_dims.x() * film_dims.y();

    rays.resize( rays_per_pass );
    pixel_ids.resize( rays_per_pass );

    const unsigned sample_index = index2DTo1D( m_current_sample, m_spp ); 

    double pixel_width  = 1.0 / static_cast<double>( film_dims.x() );
    double pixel_height = 1.0 / static_cast<double>( film_dims.y() );
    double cur_x = 0.0, cur_y = 0.0;
    unsigned ray_index = 0u;
    for( unsigned i = 0; i < film_dims.x(); ++i )
    {
        for( unsigned j = 0; j < film_dims.y(); ++j )
        {
            Vector2 pixel_coord ( Sobol::gen( sample_index, 4, ray_index),
                                  Sobol::gen( sample_index, 5, ray_index ) );
            Vector2 screen_coord( Sobol::gen( sample_index, 6, ray_index),
                                  Sobol::gen( sample_index, 7, ray_index ) );

            CameraSample sample;
            sample.screen = Vector2( cur_x + pixel_coord.x()*pixel_width,
                                     cur_y + pixel_coord.y()*pixel_height ); 
            sample.lens   = Vector2( m_rnd(), m_rnd() ); 
            sample.time   = lerp( m_time_interval.x(),
                                  m_time_interval.y(),
                                  m_rnd() );

            m_camera->generateRay( sample, rays[ ray_index ] );
            pixel_ids[ ray_index ].weight = 1.0f;
            pixel_ids[ ray_index ].pixel  = Index2( i, j );
            ray_index++;
            
            cur_y += pixel_height;

        }
        cur_x += pixel_width;
        cur_y = 0.0;
    }

    m_current_sample = index1DTo2D( sample_index+1, m_spp );
}

