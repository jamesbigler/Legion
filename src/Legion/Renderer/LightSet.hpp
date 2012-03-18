 

// Copyright (C) 2011 R. Keith Morley
//
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
// (MIT/X11 License)

/// \file LightSet.hpp
/// LightSet

#ifndef LEGION_RENDERER_LIGHTSET_HPP_
#define LEGION_RENDERER_LIGHTSET_HPP_

#include <vector>
#include <map>

namespace legion
{


class ILightShader;


class LightSet
{
public:
    LightSet();
    ~LightSet();

    void   addLight   ( const ILightShader* light );
    void   removeLight( const ILightShader* light );

    size_t numLights()const;

    void   selectLight( float rnd,
                        const ILightShader*& light,
                        float& pdf )const;

    const ILightShader* lookupLight( unsigned id )const;

private:

    /// Used for randomly choosing lights
    typedef std::vector<const ILightShader*>        LightVec;
    typedef std::map<unsigned, const ILightShader*> LightMap;

    /// Used for lookup via ID
    LightVec      m_light_vec;
    LightMap      m_light_map;
};


}


#endif // LEGION_RENDERER_LIGHTSET_HPP_
