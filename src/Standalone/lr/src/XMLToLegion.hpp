
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

#ifndef LR_XML_TO_LEGION_HPP_
#define LR_XML_TO_LEGION_HPP_

#include <rapidxml/rapidxml.hpp>
#include <Legion/Legion.hpp>
#include <memory>
#include <map>

class XMLToLegion
{
public:
    typedef rapidxml::xml_node<>       XMLNode; 
    typedef rapidxml::xml_attribute<>  XMLAttribute; 

    explicit XMLToLegion( const XMLNode* root );
    ~XMLToLegion();

private:
    typedef std::map<std::string, legion::ITexture*>  Textures;
    typedef std::map<std::string, legion::ISurface*>  Surfaces;

    void loadParams( const XMLNode* node );

    legion::IDisplay* createDisplay ( const XMLNode* display_node );
    void              createRenderer( legion::IDisplay* display,
                                      const XMLNode* renderer_node );
    void              createCamera  ( const XMLNode* camera_node );
    void              createScene   ( const XMLNode* scene_node );

    std::auto_ptr<legion::Context>   m_ctx;
    Textures                         m_textures;
    Surfaces                         m_surfaces;
    legion::Parameters               m_params;
};


#endif // LR_XML_TO_LEGION_HPP_
