
#ifndef LEGION_OBJECTS_SURFACE_LAMBERTIAN_HPP_
#define LEGION_OBJECTS_SURFACE_LAMBERTIAN_HPP_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;

class Lambertian : public ISurface
{
public:
    Lambertian( Context* context );
    ~Lambertian();
    
    void setReflectance( const Color& reflectance );
    
    const char* name()const;
    const char* sampleBSDFFunctionName()const;
    const char* evaluateBSDFFunctionName()const;
    const char* pdfFunctionName()const;
    const char* emissionFunctionName()const;

    void setVariables( const VariableContainer& container ) const ;

private:
    Color m_reflectance;
};

}

#endif // LEGION_OBJECTS_SURFACE_LAMBERTIAN_HPP_
