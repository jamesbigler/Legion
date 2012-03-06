

/// \file IFilm.hpp
/// Pure virtual interface for Film classes

#ifndef LEGION_SCENE_FILM_IFILM_HPP_
#define LEGION_SCENE_FILM_IFILM_HPP_

#include <Legion/Core/APIBase.hpp>
#include <Legion/Core/Vector.hpp>
#include <Legion/Core/Color.hpp>

namespace legion
{

class Color;

/// Interface for all Film classes
class IFilm : public APIBase
{
public:

    /// Create named Film object
                     IFilm( Context* context, const std::string& name );
    
    /// Destroy Film object
    virtual          ~IFilm();
    
    /// Set Film dimensions (width, height)
    virtual void     setDimensions( const Index2& dimensions )=0;

    /// Retrieve the current Film dimensions
    virtual Index2   getDimensions()const=0;

    /// Add a sample into the specified pixel.
    ///   \param pixel_index    Index of pixel to receive sample
    ///   \param color          Color of sample
    ///   \param weight         Weight of sample
    virtual void     addSample( const Index2& pixel_index,
                                float weight,
                                const Color& color )=0;

    /// Retrieve the value of a single pixel
    ///   \param pixel_index   Pixel position
    virtual Color    getPixel( const Index2& pixel_index )const=0;

    /// Retrieve the entire raster 
    virtual Color*   getPixels()const=0;

    /// Indicate beginning of exposure.
    /// Film should be set to a pre-render state. 
    virtual void     shutterOpen()=0;

    /// Indicate completion of exposure.
    /// Film can take any post-render actions.
    virtual void     shutterClose()=0;

    /// Indicate an intermediate pass is complete.
    /// Film can take any intermediate update steps.
    virtual void     passComplete()=0;
};


} // namespace legion

#endif // LEGION_SCENE_FILM_IFILM_H_
