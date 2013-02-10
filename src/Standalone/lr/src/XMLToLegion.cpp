
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
#include <fstream>
#include <sstream>


using namespace legion;

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
Vector3 lexical_cast<Vector3, std::string>( const std::string& arg )
{
    std::stringstream oss( arg );
    Vector3 result;
    oss >> result[0] >> result[1] >> result[2];
    return result;
}

template<>
Color lexical_cast<Color, std::string>( const std::string& arg )
{
    std::stringstream oss( arg );
    Color result;
    oss >> result[0] >> result[1] >> result[2];
    return result;
}

}


/*
const char* getAttr( const char* name, rapidxml::xml_node<>* node )
{
    rapidxml::xml_attribute<>* attr = node->first_attribute( name );
    return attr->value();
}
*/

XMLToLegion::XMLToLegion( const XMLNode* node )
    : m_ctx( new legion::Context )
{
    LEGION_ASSERT( node );

    legion::IDisplay* display = createDisplay( node->first_node("display") );
    display->beginScene( node->name() );
    createRenderer( display, node->first_node( "renderer" ) );
    createCamera  ( node->first_node( "camera" ) );
    createScene   ( node->first_node( "scene" ) );

    m_ctx->render();
}


XMLToLegion::~XMLToLegion()
{
}


void XMLToLegion::loadParams( const XMLNode* node )
{
    m_params.clear();

    for( rapidxml::xml_node<>* pnode = node->first_node();
         pnode; 
         pnode = pnode->next_sibling() )
    {
        const std::string type  = pnode->name();
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
        }
        else if( type == "vector3" )
        {
            m_params.set( name, lexical_cast<Vector3>( value ) );
        }
        else if( type == "vector4" )
        {
        }
        else if( type == "color" )
        {
            m_params.set( name, lexical_cast<Color>( value ) );
        }
        else if( type == "matrix" )
        {
        }
        else if( type == "texture" )
        {
            ITexture* texture_value = m_textures[ value ];
            m_params.set( name, texture_value );
        }
        else
        {
            throw std::runtime_error( "Unrecognized param type: " + type );
        }
    }
}


legion::IDisplay* XMLToLegion::createDisplay( const XMLNode* display_node )
{
    LEGION_ASSERT( display_node );
    XMLAttribute* attr =  display_node->first_attribute( "type" );
    std::cout << "Creating display  : '" << attr->value() << "'" << std::endl;

    const char* display_type = attr->value();
    loadParams( display_node );
    return m_ctx->createDisplay( display_type, m_params );
}


void XMLToLegion::createRenderer( IDisplay* display,
                                  const XMLNode* renderer_node )
{
    LEGION_ASSERT( display );
    LEGION_ASSERT( renderer_node );

    XMLAttribute* attr =  renderer_node->first_attribute( "type" );
    std::cout << "Creating renderer: '" << attr->value() << "'" << std::endl;

    const char* renderer_type = attr->value();
    loadParams( renderer_node );
    IRenderer* renderer = m_ctx->createRenderer( renderer_type, m_params );
    
    renderer->setDisplay( display );
    attr = renderer_node->first_attribute( "samples_per_pixel" );
    if( attr )
        renderer->setSamplesPerPixel( lexical_cast<float>( attr->value() ) );

    m_ctx->setRenderer( renderer );
}


void XMLToLegion::createCamera( const XMLNode* camera_node )
{
    LEGION_ASSERT( camera_node );
    
    XMLAttribute* attr =  camera_node->first_attribute( "type" );
    std::cout << "Creating camera  : '" << attr->value() << "'" << std::endl;

    const char* camera_type = attr->value();
    loadParams( camera_node );
    ICamera* camera = m_ctx->createCamera( camera_type, m_params );

    m_ctx->setCamera( camera );
}


void XMLToLegion::createScene( const XMLNode* scene_node )
{
    LEGION_ASSERT( scene_node );

    //
    // Create textures first since they can be referred to by surfaces 
    //
    for( const XMLNode* tex_node = scene_node->first_node( "texture" );
         tex_node;
         tex_node = tex_node->next_sibling( "texture" ) )
    {
        XMLAttribute* name_attr = tex_node->first_attribute( "name" );
        XMLAttribute* type_attr = tex_node->first_attribute( "type" );
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
        XMLAttribute* name_attr = surface_node->first_attribute( "name" );
        XMLAttribute* type_attr = surface_node->first_attribute( "type" );
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
        XMLAttribute* type_attr = geometry_node->first_attribute( "type" );
        XMLAttribute* surf_attr = geometry_node->first_attribute( "surface" );
        std::cout << "Creating geometry: '" << type_attr->value() << "'"
                  << std::endl;

        loadParams( geometry_node );
        IGeometry* geometry = 
            m_ctx->createGeometry( type_attr->value(), m_params );
        geometry->setSurface( m_surfaces[ surf_attr->value() ] );

        m_ctx->addGeometry( geometry );
    }
    
    //
    // Create environment 
    //
    {
        const XMLNode* env_node = scene_node->first_node( "environment" );
        LEGION_ASSERT( env_node );

        XMLAttribute* type_attr = env_node->first_attribute( "type" );
        std::cout << "Creating env     : '" << type_attr->value() << "'"
                  << std::endl;

        loadParams( env_node );
        m_ctx->setEnvironment( 
                m_ctx->createEnvironment( type_attr->value(), m_params )
                );
    }
}
