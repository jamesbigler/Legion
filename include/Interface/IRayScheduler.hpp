

/// \file IRayScheduler.hpp
/// Pure virtual interface for Ray Scheduler classes

#ifndef LEGION_INTERFACE_IRAYSCHEDULER_HPP_
#define LEGION_INTERFACE_IRAYSCHEDULER_HPP_

namespace legion
{

class IFilm;


class IRayScheduler
{
public:
    virtual               ~IRayScheduler();

    virtual void          setFilm( IFilm* film )=0;

    virtual void          generate( CameraSample* samples )=0;

    virtual void          update( 

  
};

}
#endif // LEGION_INTERFACE_IRAYSCHEDULER_HPP_
