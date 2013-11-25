
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

#ifndef LEGION_COMMON_UTIL_LOGGER_H_
#define LEGION_COMMON_UTIL_LOGGER_H_

#include <Legion/Common/Util/Preprocessor.hpp>
#include <iostream>
#include <ctime>
#include <sys/time.h>


#ifndef LLOG_MAX_LEVEL
#   define LLOG_MAX_LEVEL legion::Log::WARNING
#endif


#define LLOG( level )                                                          \
    if( level > LLOG_MAX_LEVEL)                         ;                      \
    else if( level > legion::Log::getReportingLevel() ) ;                      \
    else legion::Log().get( level )

#define LLOG_ERROR   LLOG( legion::Log::ERROR   )
#define LLOG_WARN    LLOG( legion::Log::WARNING )
#define LLOG_INFO    LLOG( legion::Log::INFO    )
#define LLOG_DEBUG   LLOG( legion::Log::DEBUG   )
#define LLOG_STAT    LLOG( legion::Log::STAT    )

namespace legion
{

///
/// Logger class which strives for:
///   - Efficient handling of below-threshold logging events
///   - Simplicity
///
/// Usage:
///   - Set compile-time debug level via preprocessor macro LLOG_MAX_LEVEL
///     All logging events > LLOG_MAX_LEVEL will be eliminated by compiler
///   - Set runtime debug level via Log::setReportingLevel()
///   - Use  LLOG macro to log at desired level:
///     LLOG( Log::WARNING ) << "Some warning message";
///     The logger will insert newlines after each message
///   - Defaults to writing to cerr. Use setStream() to write to cout, file, etc
///
/// Inspired by http://drdobbs.com/cpp/201804215
///
class LAPI Log
{
public:
    enum Level
    {
        ERROR=0,
        WARNING,
        STAT,
        INFO,
        DEBUG,
        DEBUG1,
        DEBUG2,
        DEBUG3,
        DEBUG4
    };

    /// INFO: Do not use this public interface directly: use LLOG macro
    LAPI Log();
    /// INFO: Do not use this public interface directly: use LLOG macro
    LAPI ~Log();
    /// INFO: Do not use this public interface directly: use LLOG macro
    LAPI std::ostream& get( Level level = INFO );

    LAPI static Level         getReportingLevel();
    LAPI static void          setReportingLevel( Level level );
    LAPI static void          setStream( std::ostream& out );

private:
    Log(const Log&);
    Log& operator=(const Log&);

    static std::string time();
    static std::string toString( Level level );
    
    static Level         s_reporting_level;
    static std::ostream* s_out;
};

}

#endif // LEGION_COMMON_UTIL_LOGGER_H_

