
#ifndef LEGION_OBJECTS_SURFACE_DIFFUSE_EMITTER_H_
#define LEGION_OBJECTS_SURFACE_DIFFUSE_EMITTER_H_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;

class DiffuseEmitter : public ISurface
{
public:
    DiffuseEmitter( Context* context );
    ~DiffuseEmitter();
    
    void setRadiance( const Color& radiance );
    
    const char* name()const;
    const char* emissionFunctionName()const;
    const char* sampleBSDFFunctionName()const;
    const char* evaluateBSDFFunctionName()const;
    const char* pdfFunctionName()const;

    void setVariables( const VariableContainer& container ) const ;

private:
    Color m_radiance;
};


}

#endif // LEGION_OBJECTS_SURFACE_DIFFUSE_EMITTER_H_
