
import bpy
import os

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


def createRenderer( xml_node ):
    pass

def createDisplay( xml_node ):
    pass

def translateCamera( blender_node, xml_node ):
    pass

def translateMesh( blender_node, xml_node ):
    pass

def translateLight( blender_node, xml_node ):
    pass

def export( context, path ):

    f = open( path, "w" )

    scene   = context.scene
    camera  = scene.camera
    objects = scene.objects
    
    for obj in objects:
        f.write( "{}: {} {}\n".format( obj.name, obj.type, obj ) )

    xml_root = ET.Element("legion_scene")
    xml_root.attrib[ "name" ] = "blender"

    xml_camera   = ET.SubElement( xml_root, "camera" )
    translateCamera( camera, xml_camera )

    xml_renderer = ET.SubElement( xml_root, "renderer" )
    createRenderer( xml_renderer )

    xml_display  = ET.SubElement( xml_root, "display" )
    createDisplay( xml_display )

    xml_scene = ET.SubElement( xml_root, "scene" )
    meshes = [ o for o in objects if o.type == 'MESH' ]
    for mesh in meshes:
        xml_mesh = ET.SubElement( xml_scene, "mesh" )
        translateMesh( mesh, xml_mesh )

    lights = [ o for o in objects if o.type == 'LAMP' ]
    for light in lights:
        xml_light = ET.SubElement( xml_scene, "light" )
        translateLight( light, xml_light )

    text  = minidom.parseString( ET.tostring( xml_root ) ).toprettyxml()

    f.write( text )
    
    return {'FINISHED'}




