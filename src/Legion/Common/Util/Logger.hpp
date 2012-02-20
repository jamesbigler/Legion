
#ifndef LEGION_COMMON_UTIL_LOGGER_H_
#define LEGION_COMMON_UTIL_LOGGER_H_


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
class Log
{
public:
    enum Level
    {
        ERROR=0,
        WARNING,
        INFO,
        DEBUG,
        DEBUG1,
        DEBUG2,
        DEBUG3,
        DEBUG4
    };

    /// INFO: Do not use this public interface directly: use LLOG macro
    Log();
    /// INFO: Do not use this public interface directly: use LLOG macro
    ~Log();
    /// INFO: Do not use this public interface directly: use LLOG macro
    std::ostream& get( Level level = INFO );

    static Level         getReportingLevel();
    static void          setReportingLevel( Level level );
    static void          setStream( std::ostream& out );

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

