
#ifndef LEGION_INTERFACE_IFILM_H_
#define LEGION_INTERFACE_IFILM_H_

#include <private/APIBase.hpp>
#include <Core/Vector.hpp>
#include <Core/Color.hpp>

namespace legion
{


/// Interface for all Film classes
class IFilm : public APIBase
{
public:

    /// Create named Film object
    explicit IFilm( const std::string& name );
    
    /// Destroy Film object
    virtual ~IFilm();
    
    /// Set Film dimensions (width, height)
    virtual void    setDimensions( const Index2& dimensions )=0;

    /// Query the current Film dimensions
    virtual Index2  getDimensions()const=0;

    /// Add a sample into the specified pixel.
    ///   \param pixel_index    Index of pixel to receive sample
    ///   \param color          Color of sample
    ///   \param weight         Weight of sample
    virtual void addSample( const Index2& pixel_index,
                            const Color& color,
                            float weight )=0;

    /// Indicate beginning of exposure.
    /// Film should be set to a pre-render state. 
    virtual void shutterOpen()=0;

    /// Indicate completion of exposure.
    /// Film can take any post-render actions.
    virtual void shutterClose()=0;

    /// Indicate an intermediate pass is complete.
    /// Film can take any intermediate update steps.
    virtual void passComplete()=0;
};


} // namespace legion

#endif // LEGION_INTERFACE_IFILM_H_
