

#include <Legion/Common/Util/Logger.hpp>

using namespace legion;

std::ostream* Log::s_out             = &std::cerr;
Log::Level    Log::s_reporting_level = Log::DEBUG;


Log::Log()
{
}


Log::~Log()
{
    *s_out << "\n";
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
        "INFO",
        "DEBUG",
        "DEBUG1",
        "DEBUG2",
        "DEBUG3",
        "DEBUG4"
    };
    return level2string[ level ];
}


std::string Log::time()
{
    char buffer[11];
    time_t t;
    std::time(&t);
    tm r = {0};
    strftime(buffer, sizeof(buffer), "%X", localtime_r(&t, &r));

    struct timeval tv;
    gettimeofday(&tv, 0);

    char result[100];
    std::sprintf(result, "%s.%03ld", buffer, (long)tv.tv_usec / 1000); 
    return result;
}
