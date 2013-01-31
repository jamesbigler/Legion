
#ifndef LEGION_OBJECTS_SURFACE_WARD_HPP_
#define LEGION_OBJECTS_SURFACE_WARD_HPP_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;

class Ward : public ISurface
{
public:
    Ward( Context* context );
    ~Ward();
    
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

#endif // LEGION_OBJECTS_SURFACE_WARD_HPP_
