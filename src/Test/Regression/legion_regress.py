#!/usr/bin/env python

import os
import os.path
import shutil
import subprocess
import pprint

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
def run_tests( legion_root, legion_bin ):
    tests = [
            ( 'dielectric'   , '-n 32' ),
            ( 'metal'        , '-n 32' ),
            ( 'monkey/monkey', '-n 8' ),
            ( 'simple'       , '-n 32' ),
            ( 'ward'         , '-n 32' ),
            ]

    lr        = os.path.join( legion_bin, 'lr' )
    scene_dir = os.path.join( legion_root, 'src/Standalone/lr/scenes' )
    dirname   = 'results_legion_regress_'
    create_dir( dirname )

    stats = []

    for test, test_args in tests:
        # execute the test
        xml_file = os.path.join( scene_dir, test+'.xml' )
        test_cmd = '{} {} {}'.format( lr, test_args, xml_file )
        print "Running <<{}>>".format( test_cmd )
        subprocess.check_call(test_cmd, shell=True, executable='/bin/bash')

        # print stats
        test_stats = [ ('name', test ) ]
        with open( 'legion.log', 'r' ) as log_file:
            for line in log_file:
                if 'STAT:' in line:
                    fields = map( lambda x: x.strip(), line.split( '|' ) )
                    if fields[1] == 'gpu':
                        gpu_name = fields[2]
                        continue
                    test_stats.append( ( fields[1], fields[2] ) )
        stats.append( test_stats )

        # copy result image to result dir
        img_file = os.path.basename( test ) + '.exr'
        os.rename( img_file, os.path.join( dirname, img_file ) ) 

    stat_filename = os.path.join( dirname, 'stats.csv' )
    with open( stat_filename, 'a' ) as stat_file:
        stat_file.write( "gpu_name = '{}'\n".format( gpu_name ) )
        stat_file.write( "stats = " )
        pp = pprint.PrettyPrinter( indent=4, stream=stat_file )
        pp.pprint( stats )


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

build_legion( "build_legion_regress_", args.legion_root, args.optix_root )
run_tests( args.legion_root, os.path.join( 'build_legion_regress_', 'bin' ) )
