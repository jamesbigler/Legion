
#ifndef LEGION_ENGINE_LIGHT_HPP_
#define LEGION_ENGINE_LIGHT_HPP_

#include <Core/Mesh.hpp>
#include <Interface/ILightShader.hpp>

namespace legion
{

    class ILightShader;
    class Mesh;

    struct Light
    {
        // TODO: Flesh this out, move to shared place if needed
        std::string    getName() 
        { 
            return shader->getName() + ":" + 
                   ( geometry ? geometry->getName() : "NULL" );
        }
        const ILightShader* shader;
        const Mesh*         geometry;
    };

}

#endif // LEGION_ENGINE_LIGHT_HPP_
