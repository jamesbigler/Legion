
#ifndef LEGION_OBJECTS_SURFACE_LAMBERTIAN_HPP_
#define LEGION_OBJECTS_SURFACE_LAMBERTIAN_HPP_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;
class ITexture;

class Lambertian : public ISurface
{
public:
    Lambertian( Context* context );
    ~Lambertian();
    
    void setReflectance( const ITexture* reflectance );
    
    const char* name()const;
    const char* sampleBSDFFunctionName()const;
    const char* evaluateBSDFFunctionName()const;
    const char* pdfFunctionName()const;
    const char* emissionFunctionName()const;

    void setVariables( VariableContainer& container ) const ;

private:
    const ITexture* m_reflectance;
};

}

#endif // LEGION_OBJECTS_SURFACE_LAMBERTIAN_HPP_
