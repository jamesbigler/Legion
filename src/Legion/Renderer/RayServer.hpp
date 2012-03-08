
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
// TODO:
//     * new design obviates templating on RSResult -- get rid of it
//     * Lots o' error checking still needed to ensure that internal state
//       is always consistent (mapped buffers, etc) and that buffers are 
//       declared.
//     * Might be better to have this class manage IO buffers completely (create
//       and destroy them).  This is necessary for most below sugs
//     * Implement double buffering output so you can be tracing while results
//       are mapped (make sure does not incur optix compile cost)
//     * Investigate multiple ray buffers so that multiple async trace() calls 
//       can be in flight at once (probably use TraceID to identify traces 
//       ( eg, TraceID trace( ); getResults( TraceID ); )
//     * Investigate clustering of trace requests if above is implemented
//     * Handle arbitrary launch dimensionality
//

#ifndef LEGION_RAYTRACER_RAYSERVER_HPP_
#define LEGION_RAYTRACER_RAYSERVER_HPP_

#include <Legion/Common/Util/Logger.hpp>

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
    void setRayBufferName   ( const std::string& name );
    void setResultBufferName( const std::string& name );

    void preprocess();

    void trace( unsigned entry_index, const std::vector<RSRay>& rays );

    void join();

    optix::Buffer  getResults();


private:
    optix::Context       m_optix_context;
    std::string          m_ray_buffer_name;
    std::string          m_result_buffer_name;

    bool                 m_results_mapped;

    boost::thread        m_thread;
    boost::mutex         m_mutex;
};


template < typename RSRay, typename RSResult >
RayServer<RSRay, RSResult>::RayServer()
    : m_results_mapped( false )
{
}


template < typename RSRay, typename RSResult >
RayServer<RSRay, RSResult>::~RayServer()
{
}


template < typename RSRay, typename RSResult >
void RayServer<RSRay, RSResult>::setContext( optix::Context context )
{
    m_optix_context = context;
}


template < typename RSRay, typename RSResult >
void RayServer<RSRay, RSResult>::setRayBufferName( const std::string& name )
{
    m_ray_buffer_name = name;
}


template < typename RSRay, typename RSResult >
void RayServer<RSRay, RSResult>::setResultBufferName( const std::string& name )
{
    m_result_buffer_name = name;
}


template < typename RSRay, typename RSResult >
void RayServer<RSRay, RSResult>::preprocess()
{
    m_optix_context->compile();
    m_optix_context->launch( 0, 0 );
}


template < typename RSRay, typename RSResult >
void RayServer<RSRay, RSResult>::trace( unsigned entry_index,
                                        const std::vector<RSRay>& rays )
{
    if( m_thread.get_id() != boost::thread().get_id() )
        throw Exception( "RayServer::trace() called twice without "
                         "calling RayServer::getResults()" );

    if( m_results_mapped )
        m_optix_context[ m_result_buffer_name ]->getBuffer()->unmap();

    // Copy ray data into buffer
    optix::Buffer ray_buffer = m_optix_context[m_ray_buffer_name]->getBuffer();
    ray_buffer->setSize( rays.size() );
    memcpy( ray_buffer->map(), &rays[0], rays.size()*sizeof( RSRay ) );
    ray_buffer->unmap();

    optix::Buffer results = m_optix_context[m_result_buffer_name]->getBuffer();
    results->setSize( rays.size() );

    // launch a thread which runs optix::Context::launch
    OptiXLaunch optix_launch( m_optix_context, entry_index, rays.size() );
    m_thread = boost::thread( optix_launch );
}


template < typename RSRay, typename RSResult >
void RayServer<RSRay, RSResult>::join()
{
    m_thread.join();
    m_thread = boost::thread();
}


template < typename RSRay, typename RSResult >
optix::Buffer RayServer<RSRay, RSResult>::getResults()
{
    join();
    optix::Buffer results = m_optix_context[m_result_buffer_name]->getBuffer();
    return results;
}


}


#endif // LEGION_RAYTRACER_RAYSERVER_HPP_

