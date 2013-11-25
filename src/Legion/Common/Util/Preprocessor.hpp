
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

/// \file Preprocessor.hpp
/// Preprocessor

#ifndef LEGION_COMMON_UTIL_PREPROCESSOR_HPP_
#define LEGION_COMMON_UTIL_PREPROCESSOR_HPP_


#ifndef LAPI                                                                    
#  if legion_EXPORTS // Set by CMAKE                                              
#    if defined( _WIN32 )                                                        
#      define LAPI __declspec(dllexport)                                        
#    elif defined( linux ) || defined ( __CYGWIN__ )                             
#      define LAPI __attribute__ ((visibility ("default")))                     
#    elif defined( __APPLE__ ) && defined( __MACH__ )                            
#      define LAPI __attribute__ ((visibility ("default")))                     
#    else                                                                        
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"                         
#    endif                                                                       
#  else
#    if defined( _WIN32 )                                                        
#      define LAPI __declspec(dllimport)                                        
#    else
#      define LAPI
#    endif
#  endif
#endif

#define LFUNC __PRETTY_FUNCTION__

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define LCUDA
#endif

#ifdef LCUDA
#    include <host_defines.h>  // For __host__ and __device__ 
#    define LHOSTDEVICE __host__ __device__
#    define LDEVICE     __device__
#else
#    define LHOSTDEVICE
#endif


#endif // LEGION_COMMON_UTIL_PREPROCESSOR_HPP_
