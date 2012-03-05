
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

//
// Notes:
//     * Implement double buffering output so you can be tracing while results
//       are mapped (make sure does not incur optix compile cost)
//     * Investigate multiple ray buffers so that multiple async trace() calls 
//       can be in flight at once (probably use TraceID to identify traces 
//       ( eg, TraceID trace( ); getResults( TraceID ); )
//     * Investigate clustering of trace requests if above is implemented
//

#ifndef LEGION_RAYTRACER_RAYSERVER_HPP_
#define LEGION_RAYTRACER_RAYSERVER_HPP_

#include <boost/thread.hpp>
#include <optixu/optixpp_namespace.h>
#include <string>

namespace legion
{


template < typename RSRay, typename RSResult >
class RayServer
{
public:
    struct OptiXLaunch
    {
        OptiXLaunch( optix::Context context,
                     unsigned entry_point_index,
                     unsigned num_rays )
            : m_context( context ),
              m_entry_point_index( entry_point_index ),
              m_num_rays( num_rays )
        {}

        void operator()()
        { m_context->launch( m_entry_point_index, m_num_rays ); }

        optix::Context m_context;
        unsigned       m_entry_point_index;
        unsigned       m_num_rays;
    };


    RayServer();
    ~RayServer();

    void setContext         ( optix::Context context );
    void setRayBufferName   ( const std::string& ray_buffer_name );
    void setResultBufferName( const std::string& result_buffer_name );

    void trace( unsigned entry_index, unsigned nrays, const RSRay* rays );

    RSResult* getResults()const;


private:
    optix::Context       m_optix_context;
    std::string          m_ray_buffer_name;
    std::string          m_result_buffer_name;

    boost::thread        m_thread;
    boost::mutex         m_mutex;
};


RayServer::RayServer()
{
}


RayServer::~RayServer()
{
}


void RayServer::setContext( optix::Context context )
{
    m_context = context;
}


void RayServer::setRayBufferName( const std::string& ray_buffer_name )
{
    m_ray_buffer_name = ray_buffer_name;
}


void RayServer::setResultBufferName( const std::string& result_buffer_name )
{
    m_result_buffer_name = result_buffer_name;
}


void RayServer::trace( unsigned entry_index, unsigned nrays, const RSRay* rays )
{

    OptiXLaunch optix_launch( m_optix_context, entry_index, nrays );
    m_thread = boost::thread( ol );
    m_thread = boost::thread( ol );
}


RSResult* RayServer::getResults()const
{
    m_thread.join();
    optix::Buffer results = m_context[ m_result_buffer_name ]->getBuffer();
}


}


#endif // LEGION_RAYTRACER_RAYSERVER_HPP_

