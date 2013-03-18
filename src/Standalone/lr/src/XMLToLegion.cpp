
// Copyright (C) 2011 R. Keith Morley 
// 
// (MIT/X11 License)
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.


#include <Legion/Legion.hpp>
#include <rapidxml/rapidxml.hpp>
#include <XMLToLegion.hpp>
#include <iostream>
#include <stdexcept>
#include <sstream>

using namespace lr;

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------
namespace
{

template<typename Target, typename Source>
Target lexical_cast( const Source& arg )
{
    std::stringstream interpreter;
    Target result;
    if( !( interpreter << arg     )       ||
        !( interpreter >> result  )       ||
        !( interpreter >> std::ws ).eof() )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
legion::Index2 lexical_cast<legion::Index2, std::string>( 
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Index2 result;
    if( !( oss >> result[0] >> result[1] ) ||
        !( oss >> std::ws ).eof()        )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
legion::Vector2 lexical_cast<legion::Vector2, std::string>( 
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Vector2 result;
    if( !( oss >> result[0] >> result[1] ) ||
        !( oss >> std::ws ).eof()        )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
legion::Vector3 lexical_cast<legion::Vector3, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Vector3 result;
    if( !( oss >> result[0] >> result[1] >> result[2] ) ||
        !( oss >> std::ws ).eof()                     )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
legion::Vector4 lexical_cast<legion::Vector4, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Vector4 result;
    if( !( oss >> result[0] >> result[1] >> result[2] >> result[3] ) ||
        !( oss >> std::ws ).eof()                                  )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
legion::Matrix lexical_cast<legion::Matrix, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Matrix result;
    if( !(oss >> result[ 0] >> result[ 1] >> result[ 2] >> result[ 3]
              >> result[ 4] >> result[ 5] >> result[ 6] >> result[ 7]
              >> result[ 8] >> result[ 9] >> result[10] >> result[11]
              >> result[12] >> result[13] >> result[14] >> result[15] ) ||
        !( oss >> std::ws ).eof()                                     )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}


template<>
legion::Color lexical_cast<legion::Color, std::string>(
        const std::string& arg )
{
    std::stringstream oss( arg );
    legion::Color result;
    if( !( oss >> result[0] >> result[1] >> result[2] ) ||
        !( oss >> std::ws ).eof()                     )
        throw std::runtime_error( "lexical_cast failure." );
    return result;
}

}

//------------------------------------------------------------------------------
//
// 
//
//------------------------------------------------------------------------------

XMLToLegion::XMLToLegion( char* text,
                          legion::Context* ctx,
                          bool create_display )
    : m_ctx( ctx ),
      m_own_context( ctx == 0 ),
      m_create_display( create_display )
{
    if( !text )
        throw std::runtime_error( "XMLToLegion: Null input text XML" );

    if( m_own_context )
        m_ctx = new legion::Context();

    try
    {
        m_doc.parse<rapidxml::parse_full>(text);
    }
    catch( rapidxml::parse_error& e )
    {
        std::cout << "XML parse error: " << e.what() 
                  << ": <" << e.where<char>() << ">" << std::endl;
        throw;
    }
    
    XMLNode* scene_node = m_doc.first_node( "legion_scene" );
    if( !scene_node )
        throw std::runtime_error( 
                "XMLToLegion: XML does not contian legion_scene"
                );

    legion::IDisplay* display = 
        createDisplay( scene_node->first_node("display") );

    if( display )
        display->beginScene( scene_node->name() );

    createRenderer( display, scene_node->first_node( "renderer" ) );
    createCamera  ( scene_node->first_node( "camera" ) );
    createScene   ( scene_node->first_node( "scene" ) );
}


XMLToLegion::~XMLToLegion()
{
    if( m_own_context )
        delete m_ctx;
}


void XMLToLegion::loadParams( const XMLNode* node )
{
    LEGION_ASSERT( node );

    m_params.clear();

    for( rapidxml::xml_node<>* pnode = node->first_node();
         pnode; 
         pnode = pnode->next_sibling() )
    {
        const std::string type  = pnode->name();
        if( !pnode->first_attribute( "name" ) )
            throw std::runtime_error( 
                    "XMLToLegion: Parameter node missing 'name' attribute"
                    );
        if( !pnode->first_attribute( "value" ) )
            throw std::runtime_error( 
                    "XMLToLegion: Parameter node missing 'value' attribute"
                    );

        const std::string name  = pnode->first_attribute( "name" )->value();
        const std::string value = pnode->first_attribute( "value" )->value();
        if( type == "string" )
        {
            m_params.set( name, value );
        }
        else if( type == "int" )
        {
            m_params.set( name, lexical_cast<int>( value ) );
        }
        else if( type == "float" )
        {
            m_params.set( name, lexical_cast<float>( value ) );
        }
        else if( type == "vector2" )
        {
            m_params.set( name, lexical_cast<legion::Vector2>( value ) );
        }
        else if( type == "vector3" )
        {
            m_params.set( name, lexical_cast<legion::Vector3>( value ) );
        }
        else if( type == "vector4" )
        {
            m_params.set( name, lexical_cast<legion::Vector4>( value ) );
        }
        else if( type == "color" )
        {
            m_params.set( name, lexical_cast<legion::Color>( value ) );
        }
        else if( type == "matrix" )
        {
            m_params.set( name, lexical_cast<legion::Matrix>( value ) );
        }
        else if( type == "texture" )
        {
            legion::ITexture* texture_value = 0; 
            const XMLAttribute* type_attr = pnode->first_attribute( "type" );
            if( type_attr )
            {
                // We have an inline declared constant texture
                const std::string type  = type_attr->value();
                const std::string value = pnode->first_attribute( "value" )->value();
                
                legion::Parameters tparams;
                if( type == "float" )
                    tparams.set( "value", lexical_cast<float>( value ) );
                else if( type == "vector2" )
                    tparams.set( "value", lexical_cast<legion::Vector2>( value ) );
                else if( type == "vector3" )
                    tparams.set( "value", lexical_cast<legion::Vector2>( value ) );
                else if( type == "color" )
                    tparams.set( "value", lexical_cast<legion::Color>( value ) );

                texture_value = m_ctx->createTexture( "ConstantTexture",  tparams );
            }
            else
            {
                texture_value = m_textures[ value ];
            }
            if( !texture_value )
                throw std::runtime_error( "XMLToLegion: Unknown texture "
                                          "referenced '" + value + "'" );

            m_params.set( name, texture_value );
        }
        else if( type == "surface" )
        {
            legion::ISurface* surface_value = m_surfaces[ value ];
            if( !surface_value )
                throw std::runtime_error( "XMLToLegion: Unknown surface "
                                          "referenced '" + value + "'" );

            m_params.set( name, surface_value );
        }
        else
        {
            throw std::runtime_error( "Unrecognized param type: " + type );
        }
    }
}


legion::IDisplay* XMLToLegion::createDisplay( const XMLNode* display_node )
{
    if( !m_create_display )
        return 0;
        
    if( !display_node )
        throw std::runtime_error( "XMLToLegion: XML file missing display node");

    const XMLAttribute* attr =  display_node->first_attribute( "type" );
    if( !attr )
        throw std::runtime_error(
                "XMLToLegion: Display node missing 'type' attribute"
                );

    std::cout << "Creating display  : '" << attr->value() << "'" << std::endl;

    const char* display_type = attr->value();
    loadParams( display_node );
    return m_ctx->createDisplay( display_type, m_params );
}


void XMLToLegion::createRenderer( legion::IDisplay* display,
                                  const XMLNode* renderer_node )
{
    if( !renderer_node )
        throw std::runtime_error(
                "XMLToLegion: XML file missing renderer node"
                );

    const XMLAttribute* attr =  renderer_node->first_attribute( "type" );
    if( !attr )
        throw std::runtime_error(
                "XMLToLegion: Renderer node missing 'type' attribute"
                );
    std::cout << "Creating renderer: '" << attr->value() << "'" << std::endl;

    const char* renderer_type = attr->value();
    loadParams( renderer_node );
    legion::IRenderer* renderer =
        m_ctx->createRenderer( renderer_type, m_params );
    

    if( display )
        renderer->setDisplay( display );

    attr = renderer_node->first_attribute( "samples_per_pixel" );
    if( attr )
        renderer->setSamplesPerPixel( lexical_cast<float>( attr->value() ) );

    attr = renderer_node->first_attribute( "resolution" );
    if( attr )
        renderer->setResolution( 
                lexical_cast<legion::Index2>( std::string( attr->value() ) ) );

    m_ctx->setRenderer( renderer );
}


void XMLToLegion::createCamera( const XMLNode* camera_node )
{
    if( !camera_node )
        throw std::runtime_error(
                "XMLToLegion: XML file missing camera node"
                );
    
    const XMLAttribute* attr =  camera_node->first_attribute( "type" );
    if( !attr )
        throw std::runtime_error(
                "XMLToLegion: Camera node missing 'type' attribute"
                );
    std::cout << "Creating camera  : '" << attr->value() << "'" << std::endl;
    

    const char* camera_type = attr->value();
    loadParams( camera_node );
    legion::ICamera* camera = m_ctx->createCamera( camera_type, m_params );
    attr = camera_node->first_attribute( "camera_to_world" );
    if( attr )
    {
        legion::Matrix m = lexical_cast<legion::Matrix>(
                std::string( attr->value() )
                );
        camera->setCameraToWorld( m ); 
        std::cerr << "setting camera matrix to " << m << std::endl;
    }

    m_ctx->setCamera( camera );
}


void XMLToLegion::createScene( const XMLNode* scene_node )
{
    if( !scene_node )
        throw std::runtime_error(
                "XMLToLegion: XML file missing scene node"
                );

    //
    // Create textures first since they can be referred to by surfaces 
    //
    for( const XMLNode* tex_node = scene_node->first_node( "texture" );
         tex_node;
         tex_node = tex_node->next_sibling( "texture" ) )
    {
        const XMLAttribute* name_attr = tex_node->first_attribute( "name" );
        const XMLAttribute* type_attr = tex_node->first_attribute( "type" );
        if( !name_attr )
            throw std::runtime_error( 
                    "XMLToLegion: Texture node missing 'name' attribute"
                    );
        if( !type_attr )
            throw std::runtime_error( 
                    "XMLToLegion: Texture node missing 'type' attribute"
                    );

        std::cout << "Creating texture : '" << type_attr->value() << "'"
                  << " : '" << name_attr->value() << "'" << std::endl;
        loadParams( tex_node );

        m_textures.insert( 
                std::make_pair( 
                    std::string( name_attr->value() ),
                    m_ctx->createTexture( type_attr->value(), m_params )
                    )
                );
    }

    //
    // Create surfaces next since they can be referred to by geometries 
    //
    for( const XMLNode* surface_node = scene_node->first_node( "surface" );
         surface_node;
         surface_node = surface_node->next_sibling( "surface" ) )
    {
        const XMLAttribute* name_attr = surface_node->first_attribute( "name" );
        const XMLAttribute* type_attr = surface_node->first_attribute( "type" );
        if( !name_attr )
            throw std::runtime_error( 
                    "XMLToLegion: Surface node missing 'name' attribute"
                    );
        if( !type_attr )
            throw std::runtime_error( 
                    "XMLToLegion: Surface node missing 'type' attribute"
                    );

        std::cout << "Creating surface : '" << type_attr->value() << "'"
                  << " : '" << name_attr->value() << "'" << std::endl;

        loadParams( surface_node );
        m_surfaces.insert( 
                std::make_pair( 
                    std::string( name_attr->value() ),
                    m_ctx->createSurface( type_attr->value(), m_params )
                    )
                );
    }
    
    //
    // Create geometries 
    //
    for( const XMLNode* geometry_node = scene_node->first_node( "geometry" );
         geometry_node;
         geometry_node = geometry_node->next_sibling( "geometry" ) )
    {
        const XMLAttribute* type_attr =
            geometry_node->first_attribute( "type" );

        const XMLAttribute* surf_attr =
            geometry_node->first_attribute( "surface" );

        if( !surf_attr )
            throw std::runtime_error( 
                    "XMLToLegion: Geometry node missing 'surface' attribute"
                    );
        if( !type_attr )
            throw std::runtime_error( 
                    "XMLToLegion: Geometry node missing 'type' attribute"
                    );

        const XMLAttribute* name_attr = geometry_node->first_attribute("name");
        const std::string name = name_attr ? name_attr->value() : "";
        std::cout << "Creating geometry: " << type_attr->value() << ": '"
                  << name << "'" << std::endl;

        loadParams( geometry_node );
        legion::IGeometry* geometry = 
            m_ctx->createGeometry( type_attr->value(), m_params );

        legion::ISurface* surface = m_surfaces[ surf_attr->value() ];
        if( !surface )
            throw std::runtime_error( 
                    "XMLToLegion: Geometry node refers to unknown surface '" +
                    std::string( surf_attr->value() ) + "'"
                    );

        geometry->setSurface( surface );

        m_ctx->addGeometry( geometry );
    }
    
    //
    // Create environment 
    //
    const XMLNode* env_node = scene_node->first_node( "environment" );
    if( env_node )
    {
        const XMLAttribute* type_attr = env_node->first_attribute( "type" );
        if( !type_attr )
            throw std::runtime_error( 
                    "XMLToLegion: Environment  node missing 'type' attribute"
                    );

        std::cout << "Creating env     : '" << type_attr->value() << "'"
                  << std::endl;

        loadParams( env_node );
        m_ctx->setEnvironment( 
                m_ctx->createEnvironment( type_attr->value(), m_params )
                );
    }
}
