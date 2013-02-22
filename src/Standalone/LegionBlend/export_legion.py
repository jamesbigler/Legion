
import bpy
import os
import os.path

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
        pass


    def createDisplay( self, xml_node ):
        pass


    def translateCamera( self, blender_node, xml_node ):
        pass


    def translateMesh( self, blender_node, xml_node ):
        xml_node.attrib[ "type" ] = "TriMesh" 
        xml_node.attrib[ "name" ] = blender_node.name
        filename_param = ET.SubElement( xml_node, "string" )
        datafile = blender_node.name + ".lmesh"
        filename_param.attrib[ "name"  ] = "datafile" 
        filename_param.attrib[ "value" ] = datafile 

        mesh = blender_node.data
        self.xml_file.write( "Creating {}\n".format( datafile ) )
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
        objects = scene.objects
        objects = scene.objects
        
        cam = camera.data
        self.xml_file.write( "{}\n".format( cam ) )
        self.xml_file.write( "\t{}\n".format( camera.matrix_world ) )

        for obj in objects:
            self.xml_file.write( "{} {}\n".format( obj.type, obj ) )

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




