 

// Copyright (C) 2011 R. Keith Morley
//
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
// (MIT/X11 License)

/// \file AutoTimerHelpers.hpp
/// AutoTimerHelpers

#ifndef LEGION_AUTOTIMERHELPERS_HPP_
#define LEGION_AUTOTIMERHELPERS_HPP_

#include <Legion/Common/Util/Logger.hpp>
#include <Legion/Common/Util/Timer.hpp>

namespace legion
{

struct LoopTimerInfo
{
    LoopTimerInfo( const std::string& name )
        : name( name ),
          iterations( 0u ),
          max_time( 0.0 ),
          total_time( 0.0 )
    {}

    void operator()( double time_elapsed )
    {
        ++iterations;
        max_time = std::max( max_time, time_elapsed );
        total_time += time_elapsed;
    }

    void log()
    {
        LLOG_INFO << std::fixed
                  << name 
                  << " sum: " << total_time 
                  << " max: " << max_time
                  << " avg: " << averageTime();
    }

    void reset()
    {
        iterations = 0u;
        max_time   = 0.0;
        total_time = 0.0;
    }

    double averageTime()const
    {
        return total_time / static_cast<double>( iterations );
    }

    std::string name;
    unsigned    iterations;
    double      max_time;
    double      total_time;
};



struct PrintTimeElapsed
{
    PrintTimeElapsed( const char* event ) : m_event( event ) {}

    void operator()( double time_elapsed )
    { LLOG_INFO << m_event << ": " << time_elapsed << "s"; }

    std::string m_event;
};

typedef AutoTimer<PrintTimeElapsed> AutoPrintTimer;


}


#endif // LEGION_AUTOTIMERHELPERS_HPP_
