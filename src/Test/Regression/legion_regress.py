#!/usr/bin/env python

import os
import os.path
import shutil
import subprocess
import pprint
from termcolor import colored

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


    make_cmd = 'make -j4 lr'
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
            ( 'dielectric'   , '-n 4096 -r 512 384 -s 4 -d 1' ),
            ( 'metal'        , '-n 1024 -r 512 384 -s 3 -d 1' ),
            ( 'monkey/monkey', '-n 1024 -r 480 270 -s 2 -d 1' ),
            ( 'simple'       , '-n 512  -r 512 384 -d 4'  ),
            ( 'ward'         , '-n 1024 -r 512 384 -s 3 -d 1' ),
            ]

<<<<<<< HEAD
=======
    lr        = os.path.join( legion_bin, 'lr' )
    scene_dir = os.path.join( legion_root, 'src/Standalone/lr/scenes' )
>>>>>>> 6c5b5c4f10e51ee66b736fc0e3e2c2c98a4a4f33
    create_dir( dirname )

    stats = {} 

    os.environ[ 'optix.compile.doRematerialization' ] = '100'
    for test, test_args in tests:
        # execute the test
        xml_file = os.path.join( scene_dir, test+'.xml' )
        test_cmd = '{} {} {}'.format( lr, test_args, xml_file )
        print "Running <<{}>>".format( test_cmd )
        subprocess.check_call(test_cmd, shell=True, executable='/bin/bash')

        # print stats
        test_stats = {}
        with open( 'legion.log', 'r' ) as log_file:
            for line in log_file:
                if 'STAT:' in line:
                    fields = map( lambda x: x.strip(), line.split( '|' ) )
                    if fields[1] == 'gpu':
                        gpu_name = fields[2].strip( ',' )
                        continue
                    test_stats[ fields[1] ] = fields[2]
        stats[test] = test_stats

        # copy result image to result dir
        img_file = os.path.basename( test ) + '.exr'
        os.rename( img_file, os.path.join( dirname, img_file ) ) 

    stat_filename = os.path.join( dirname, 'stats.py' )
    with open( stat_filename, 'a' ) as stat_file:
        stat_file.write( "gpu_name = '{}'\n".format( gpu_name ) )
        stat_file.write( "stats = " )
        pp = pprint.PrettyPrinter( indent=4, stream=stat_file )
        pp.pprint( stats )


#------------------------------------------------------------------------------
#
#
#
#------------------------------------------------------------------------------
import itertools
import math

def compare_images( name, gpixels, rpixels, tol_avg, tol_max, max_bad_pixels ):
    if len( gpixels ) != len( rpixels ):
        raise AssertionError( "compare_images given images with diff size" )

    def luminance( p ):
        return 0.30*p[0] + 0.59*p[1] + 0.11*p[2] 


    def pixel_diff( p0, p1 ):
        if( p0[0] == p1[0] and 
            p0[1] == p1[1] and
            p0[2] == p1[2] ):
            return 0.0
        pdiff = ( math.fabs( p1[0]-p0[0] ),
                  math.fabs( p1[1]-p0[1] ),
                  math.fabs( p1[2]-p0[2] ) )
        pdiff_lum = luminance( pdiff )
        avg_lum    = ( luminance( p0 ) + luminance( p1 ) ) * 0.5
        return pdiff_lum / avg_lum

    num_bad_pixels = 0
    diff_sum = 0.0
    for gpixel, rpixel in itertools.izip( gpixels, rpixels ):
        diff = pixel_diff( gpixel, rpixel )
        diff_sum += diff
        if diff > tol_max:
            num_bad_pixels += 1
    diff_avg = diff_sum / float( len( gpixels ) )

    if diff_avg > tol_avg or num_bad_pixels > max_bad_pixels:
        print "[FAIL] {} avg pixel difference:{} bad pixel count {}".format(
                name, diff_avg, num_bad_pixels )
        return False
    else:
        print "[PASS] {} avg pixel difference:{} bad pixel count {}".format(
                name, diff_avg, num_bad_pixels )
        return True 


#------------------------------------------------------------------------------
#
#
#
#------------------------------------------------------------------------------

import OpenEXR
import Imath 
import glob
import array
import sys

def check_results( gold_dir, results_dir, idiff_path ):

    gold_files   = glob.glob( os.path.join( gold_dir, '*.exr' ) )
    result_files = glob.glob( os.path.join( results_dir, '*.exr' ) )

    failed = []
    for gold_file in gold_files:
        basename = os.path.basename( gold_file )
        result_file = '' 
        for file in result_files:
            if os.path.basename( file ) == basename:
                result_files.remove( file )
                result_file = file
                break
        if not result_file:
            print colored(
                    "WARNING: No result found to match '{}'".format(
                        gold_file ), 'yellow'
                    )
            failed.append( gold_file + ': (No matching result file)' )
            continue
        print colored( "Comparing '{}':'{}'".format( gold_file, result_file ),
                       'blue' )
        if idiff_path:
            idiff_cmd = idiff_path + ' -v {} {}'.format(gold_file, result_file)
            res = subprocess.call(idiff_cmd, shell=True, executable='/bin/bash')
            if res != 0:
                print colored( "TEST FAILED", 'red' )
                failed.append( result_file + ': (Image compare fail)' )
            else:
                print colored( "TEST PASSED", 'green' )


    for result_file in result_files:
        print colored(
                "WARNING: No gold found to match '{}' ... ignoring".format(
                    result_file), 'yellow' 
                )

    num_passed = len(gold_files) - len(failed)
    num_tests  = len(gold_files)
    color = 'green' if num_passed == num_tests else 'red' 
    print colored( "[{}/{}] tests passed.".format(num_passed, num_tests), color)
    if len(failed):
        print "Failure cases:"
        for fail_file in failed:
            print colored( "\t{}".format( fail_file ), 'yellow' )
        
    import legion_regress_gold.stats as gold_stats
    sys.path.insert( 0, results_dir)
    import stats as results_stats

    print colored( 'Comparing render times (seconds)', 'blue' )
    for name, gold_results in gold_stats.stats[results_stats.gpu_name].items():
        print '{:<40}'.format( colored( name, 'yellow' ) ),
        test_results = results_stats.stats[ name ]
        gold_render_time = float( gold_results[ 'render time' ] )
        test_render_time = float( test_results[ 'render time' ] )
        speedup = test_render_time / gold_render_time
        color = 'green' if speedup > 1.05 else 'red' if speedup < 0.95 else 'white'
        print colored( '{:>8.3f} -> {:>8.3f} : {:>5.3f}'.format(
            gold_render_time,
            test_render_time,
            speedup ), color )



import argparse
parser = argparse.ArgumentParser()
parser.add_argument( 'legion_root', 
                     metavar='LEGION_ROOT_DIR',
                     default='.',
                     help='Path to legion root directory ' )

parser.add_argument( '-o', '--optix-root', 
                     metavar='OPTIX_ROOT_DIR',
                     default='.',
                     help='Path to optix root directory ' )

parser.add_argument( '-i', '--idiff-path', 
                     metavar='IDIFF_PATH',
                     default='./extern/idiff',
                     help='Path to idiff executable' )

args = parser.parse_args()

build_legion( "build_legion_regress_",
              args.legion_root,
              args.optix_root )

run_tests( 'results_legion_regress_',
           args.legion_root,
           os.path.join( 'build_legion_regress_', 'bin' ) )
<<<<<<< HEAD
    
=======

check_results( 'legion_regress_gold',
               'results_legion_regress_',
               args.idiff_path ) 
>>>>>>> 6c5b5c4f10e51ee66b736fc0e3e2c2c98a4a4f33
