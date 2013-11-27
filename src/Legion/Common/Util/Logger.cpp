

#include <Legion/Common/Util/Logger.hpp>
#include <cstdio>
#include <ctime>
#if !defined(_WIN32)
#  include <sys/time.h>
#endif

using namespace legion;

std::ostream* Log::s_out             = &std::cerr;
Log::Level    Log::s_reporting_level = Log::DEBUG;


Log::Log()
{
}


Log::~Log()
{
    *s_out << std::endl;
}


std::ostream& Log::get( Log::Level level )
{
    *s_out << "[" << time() << "] " << toString( level ) << ": "
           << std::string( (level > INFO ? level - INFO : 0 ), '\t' );
    return *s_out;
}


Log::Level Log::getReportingLevel()
{
    return s_reporting_level;
}


void Log::setReportingLevel( Level level )
{
    s_reporting_level = level;
}

void Log::setStream( std::ostream& out )
{
    s_out = &out;
}

std::string Log::toString( Log::Level level )
{
    static const char* const level2string[] = 
    {
        "ERROR",
        "WARNING",
        "STAT",
        "INFO",
        "DEBUG",
        "DEBUG1",
        "DEBUG2",
        "DEBUG3",
        "DEBUG4"
    };
    return level2string[ level ];
}

#if defined(_WIN32)

// http://social.msdn.microsoft.com/Forums/vstudio/en-US/430449b3-f6dd-4e18-84de-eebd26a8d668/gettimeofday?forum=vcgeneral
// correction: http://stackoverflow.com/questions/1676036/what-should-i-use-to-replace-gettimeofday-on-windows

#include <Legion/Common/Util/Assert.hpp>
#include <time.h>
#include <windows.h>
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  116444736000000000Ui64 // CORRECT
#else
  #define DELTA_EPOCH_IN_MICROSECS  116444736000000000ULL // CORRECT
#endif
 
struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
 
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;
 
  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);
 
    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;
 
    /*converting file time to unix epoch*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS; 
    tmpres /= 10;  /*convert into microseconds*/
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }
 
  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
    long timezone_val;
    LEGION_ASSERT(_get_timezone(&timezone_val) == 0);
    tz->tz_minuteswest = timezone_val / 60;
    LEGION_ASSERT(_get_daylight(&tz->tz_dsttime) == 0);
  }
 
  return 0;
}
#endif

std::string Log::time()
{
    char buffer[11];
    time_t t;
    std::time(&t);
    tm r; // ={0};
#if defined(_WIN32)
    LEGION_ASSERT(localtime_s(&r, &t) == 0);
    strftime(buffer, sizeof(buffer), "%X", &r);
#else
    strftime(buffer, sizeof(buffer), "%X", localtime_r(&t, &r));
#endif

    struct timeval tv;
    gettimeofday(&tv, 0);

    char result[100];
#if defined (_WIN32)
    _snprintf_s(result, sizeof(result), _TRUNCATE, "%s.%03ld", buffer, (long)tv.tv_usec / 1000); 
#else
    std::snprintf(result, sizeof(result)-1, "%s.%03ld", buffer, (long)tv.tv_usec / 1000); 
    result[99] = '\0';
#endif
    return result;
}
