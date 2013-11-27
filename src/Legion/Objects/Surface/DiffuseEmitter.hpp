
#ifndef LEGION_OBJECTS_SURFACE_DIFFUSE_EMITTER_H_
#define LEGION_OBJECTS_SURFACE_DIFFUSE_EMITTER_H_


#include <Legion/Objects/Surface/ISurface.hpp>
#include <Legion/Common/Util/Preprocessor.hpp>
#include <Legion/Core/Color.hpp>


namespace legion
{

class VariableContainer;

class LCLASSAPI DiffuseEmitter : public ISurface
{
public:
    LAPI static ISurface* create( Context* context, const Parameters& params );

    LAPI DiffuseEmitter( Context* context );
    LAPI ~DiffuseEmitter();
    
    LAPI void setRadiance( const Color& radiance );
    
    LAPI const char* name()const;
    LAPI const char* emissionFunctionName()const;
    LAPI const char* sampleBSDFFunctionName()const;
    LAPI const char* evaluateBSDFFunctionName()const;
    LAPI const char* pdfFunctionName()const;

    LAPI void setVariables( VariableContainer& container ) const ;

private:
    Color m_radiance;
};


}

#endif // LEGION_OBJECTS_SURFACE_DIFFUSE_EMITTER_H_
