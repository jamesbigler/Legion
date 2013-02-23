
import bpy
import os
import os.path
import math 

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

class Exporter:

    def __init__( self, context, filepath ):
        self.context = context
        self.filepath = filepath
        self.dirpath  = os.path.dirname( filepath )
        self.xml_file = open( self.filepath, "w" )

    ############################################################################
    #
    # Helper functions 
    #
    ############################################################################
    def createRenderer( self, xml_node ):
        scene   = self.context.scene
        render  = scene.render

        xml_node.attrib[ "type"              ] = "ProgressiveRenderer"
        xml_node.attrib[ "samples_per_pixel" ] = "32"
        xml_node.attrib[ "resolution"        ] = "{} {}".format( render.resolution_x, render.resolution_y )

        pass


    def createDisplay( self, xml_node ):
        pass


    def translateCamera( self, blender_node, xml_node ):
        camera = blender_node.data
        matrix = blender_node.matrix_world
        
        xml_node.attrib[ "type" ] = "ThinLens"

        matrix_string = \
                "{} {} {} {}  {} {} {} {}  {} {} {} {}  {} {} {} {}".format( 
                        matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
                        matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
                        matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3],
                        matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3] )
        xml_node.attrib[ "camera_to_world" ] = matrix_string 

        focal_dist = camera.dof_distance
        filename_param = ET.SubElement( xml_node, "float" )
        filename_param.attrib[ "name"  ] = "focal_distance" 
        filename_param.attrib[ "value" ] = "{}".format( focal_dist ) 

        #h_offset   = math.tan( camera.angle_x ) * focal_dist * 0.5
        h_offset   = math.tan( camera.angle_x*0.5 ) * focal_dist
        v_offset   = math.tan( camera.angle_y*0.5 ) * focal_dist
        lrbt_string = "{} {} {} {}".format( 
                -h_offset, h_offset, -v_offset, v_offset
                )
        filename_param = ET.SubElement( xml_node, "vector4" )
        filename_param.attrib[ "name"  ] = "view_plane" 
        filename_param.attrib[ "value" ] = lrbt_string 

        filename_param = ET.SubElement( xml_node, "float" )
        filename_param.attrib[ "name"  ] = "aperture_radius" 
        filename_param.attrib[ "value" ] = "{}".format( 
                camera.cycles.aperture_size )



    def translateMesh( self, blender_node, xml_node ):

        xml_node.attrib[ "type" ] = "TriMesh" 
        xml_node.attrib[ "name" ] = blender_node.name
        filename_param = ET.SubElement( xml_node, "string" )
        datafile = blender_node.name + ".lmesh"
        filename_param.attrib[ "name"  ] = "datafile" 
        filename_param.attrib[ "value" ] = datafile 

        mesh = blender_node.to_mesh( self.context.scene, True, 'RENDER' )
        mesh.calc_normals()
        mesh.transform( blender_node.matrix_world )

        with open( os.path.join( self.dirpath, datafile ), "w" ) as df:
            verts = mesh.vertices
            mats  = mesh.materials
            df.write( "vertcount {}\n".format( len(verts) ) )
            for vert in verts:
                df.write( "{} {} {} {} {} {}\n".format( 
                    vert.co[0],     vert.co[1],     vert.co[2],
                    vert.normal[0], vert.normal[1], vert.normal[2] ) 
                    )

            polygons = mesh.polygons
            df.write( "polycount {}\n".format( len( polygons ) ) )
            for poly in mesh.polygons:
                df.write( "{}".format( poly.loop_total ) )
                for loop_index in range( poly.loop_start,
                                         poly.loop_start + poly.loop_total ):
                    df.write(" {}".format( 
                        mesh.loops[loop_index].vertex_index )
                        )
                df.write( "\n" )


    def translateLight( self, blender_node, xml_node ):
        pass


    def gatherMaterials( self, mesh_objs ):
        mats = set() 
        for me in mesh_objs:
            mesh = me.data
            for mat in mesh.materials:
                mats.add( mat )
        return list( mats )


    def gatherTextures( self, material_objs ):
        textures = set() 
        for mat in material_objs:
            for tex_slot in mat.texture_slots:
                if tex_slot:
                    textures.add( tex_slot )
        return list( textures )


    ############################################################################
    #
    # Entry point for export
    #
    ############################################################################
    def export( self ):

        scene   = self.context.scene
        camera  = scene.camera
        objects = [ obj for obj in scene.objects if obj.is_visible( scene ) ]

        xml_root = ET.Element("legion_scene")
        xml_root.attrib[ "name" ] = "blender"

        xml_camera   = ET.SubElement( xml_root, "camera" )
        self.translateCamera( camera, xml_camera )

        xml_renderer = ET.SubElement( xml_root, "renderer" )
        self.createRenderer( xml_renderer )

        xml_display  = ET.SubElement( xml_root, "display" )
        self.createDisplay( xml_display )

        xml_scene = ET.SubElement( xml_root, "scene" )
        meshes = [ o for o in objects if o.type == 'MESH' ]
        
        materials = self.gatherMaterials( meshes )
        textures  = self.gatherTextures( materials )

        for mat in materials:
            self.xml_file.write( "Material: {}\n".format( mat ) )
        for tex in textures:
            self.xml_file.write( "Texture: {}\n".format( tex) )

        for mesh in meshes:
            xml_mesh = ET.SubElement( xml_scene, "geometry" )
            self.translateMesh( mesh, xml_mesh )

        lights = [ o for o in objects if o.type == 'LAMP' ]
        for light in lights:
            xml_light = ET.SubElement( xml_scene, "light" )
            self.translateLight( light, xml_light )

        text  = minidom.parseString( ET.tostring( xml_root ) ).toprettyxml()

        self.xml_file.write( text )
        
        return {'FINISHED'}




