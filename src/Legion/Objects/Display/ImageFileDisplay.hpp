
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

#ifndef LEGION_OBJECTS_DISPLAY_IMAGE_FILE_DISPLAY_HPP_
#define LEGION_OBJECTS_DISPLAY_IMAGE_FILE_DISPLAY_HPP_

#include <Legion/Objects/Display/IDisplay.hpp>
#include <Legion/Common/Util/Timer.hpp>
#include <string>

namespace legion
{

class VariableContainer;

class ImageFileDisplay : public IDisplay
{
public:
    static IDisplay* create( Context* context, const Parameters& params );

    ImageFileDisplay( Context* context, const char* filename );

    void beginScene    ( const std::string& scene_name );
    void setUpdateCount( unsigned m_update_count );
    void beginFrame    ();
    void updateFrame   ( const Index2& resolution, const float* pixels );
    void completeFrame ( const Index2& resolution, const float* pixels );
private:
    static const int  s_field_width = 28;

    std::string       m_scene_name;
    unsigned          m_update_count;
    unsigned          m_cur_update;
    std::string       m_filename;
    Timer             m_preprocess_timer;
    Timer             m_render_timer;
    Timer             m_postprocess_timer;
};

}

#endif // LEGION_OBJECTS_DISPLAY_IMAGE_FILE_DISPLAY_HPP_
