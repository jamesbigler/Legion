#!/usr/bin/env python

import os
import os.path
import shutil
import subprocess

def create_dir( dirname, remove_preexisting=True ):
    if remove_preexisting and os.path.exists( dirname ):
        print "Removing directory '{}' ...".format( dirname ),
        shutil.rmtree( dirname )
        print "done"
    print "Creating directory '{}' ...".format( dirname ),
    os.mkdir( dirname )
    print "done"

#------------------------------------------------------------------------------
#
#
#
#------------------------------------------------------------------------------
def build_legion( dirname, legion_root, optix_root ):

    create_dir( dirname )
    os.chdir( dirname )
    if not os.path.isabs( legion_root ):
        legion_root = os.path.join( '..', legion_root )
        optix_root  = os.path.join( '..', optix_root )

    optix_inc = os.path.join( optix_root, 'include' )
    optix_lib = os.path.join( optix_root, 'build_release/lib/liboptix.1.dylib' )
    cmake_cmd = 'cmake -DOptiX_INCLUDE:PATH={} -Doptix_LIBRARY={} {}'.format(
            optix_inc,
            optix_lib,
            legion_root
            )

    print "Running <<{}>>".format( cmake_cmd )
    subprocess.check_call( cmake_cmd, shell=True, executable='/bin/bash' )


    make_cmd = 'make -j4'
    print "Running <<{}>>".format( make_cmd )
    subprocess.check_call( make_cmd, shell=True, executable='/bin/bash' )
    
    os.chdir( '..' )


#------------------------------------------------------------------------------
#
#
#
#------------------------------------------------------------------------------
def run_tests( dirname, legion_root, legion_bin ):
    tests = [
            'dielectric',
            'metal', 
            'monkey/monkey',
            'simple',
            'ward',
            ]

    create_dir( dirname )

    lr = os.path.join( legion_bin, 'lr' )
    scene_dir = os.path.join( legion_root, 'src/Standalone/lr/scenes' )
    for test in tests:
        xml_file = os.path.join( scene_dir, test+'.xml' )
        test_cmd = '{} {}'.format( lr, xml_file )
        print "Running <<{}>>".format( test_cmd )
        subprocess.check_call( test_cmd, shell=True, executable='/bin/bash' )

        img_file = os.path.basename( test ) + '.exr'
        os.rename( img_file, os.path.join( dirname, img_file ) ) 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( 'legion_root', 
                     metavar='LEGION_ROOT_DIR',
                     default='.',
                     help='Directory path to legion root directory ' )

parser.add_argument( '-o', '--optix-root', 
                     metavar='OPTIX_ROOT_DIR',
                     default='.',
                     help='Directory path to optix root directory ' )

args = parser.parse_args()

build_legion( "build_legion_regress_",
              args.legion_root,
              args.optix_root )

run_tests( 'results_legion_regress_',
           args.legion_root,
           os.path.join( 'build_legion_regress_', 'bin' ) )
    
