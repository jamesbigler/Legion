
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

#include <Legion/Renderer/OptiXScene.hpp>
#include <Legion/Common/Util/Image.hpp>
#include <config.hpp>

using namespace legion;


namespace
{
    std::string ptxFilename( const std::string& cuda_filename )
    {
        return PTX_DIR + "/cuda_compile_ptx_generated_" + cuda_filename + ".ptx";
    }
}


OptiXScene::OptiXScene()
{
    m_optix_context = optix::Context::create();
    m_camera        = m_optix_context->createProgramFromPTXFile(
                            ptxFilename( "camera.cu" ), "legionCamera" );

    m_optix_context->setEntryPointCount( 1 );
    m_optix_context->setRayGenerationProgram( 0, m_camera );

    m_output_buffer = m_optix_context->createBuffer( 
                          RT_BUFFER_OUTPUT,
                          RT_FORMAT_FLOAT4,
                          512u, 512u );

    m_optix_context[ "output_buffer" ]->set( m_output_buffer );
}


OptiXScene::~OptiXScene()
{

}


optix::Buffer OptiXScene::getOutputBuffer()
{
    return m_output_buffer;
}


void OptiXScene::renderPass()
{
    m_optix_context->launch( 0, 512, 512 );
    const std::string filename = "test.exr";
    writeOpenEXR( filename, 512, 512, 4,
                  static_cast<float*>( m_output_buffer->map() ) );
    m_output_buffer->unmap();
}

