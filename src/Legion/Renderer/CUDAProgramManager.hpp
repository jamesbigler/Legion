
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

#ifndef LEGION_RENDERER_CUDA_PROGRAM_MANAGER_HPP_
#define LEGION_RENDERER_CUDA_PROGRAM_MANAGER_HPP_

#include <optixu/optixpp_namespace.h>
#include <map>

namespace legion
{

class CUDAProgramManager
{
public:
    CUDAProgramManager();
    ~CUDAProgramManager();

    optix::Program load( const std::string& name,
                         const std::string& cuda_filename = "",
                         const std::string& cuda_function_name = "" );

    optix::Program get( const std::string& name ) const;

private:
    typedef std::map< std::string, optix::Program> Registry;
    Registry m_registry;
};

}


#endif // LEGION_RENDERER_CUDA_PROGRAM_MANAGER_HPP_
