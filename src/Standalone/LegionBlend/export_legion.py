
import bpy
import os
import os.path
import math 

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import struct 
import array 
import mathutils


class NodeTree:
    class Node:
        def __init__( self, blender_node ):
            self.blender_node = node
            self.inputs  = []
            self.outputs = []
    def __init__( self, outfile, blender_node_tree ):
        outfile.write( "Node tree '{}':\n".format( blender_node_tree.name ) )
        for node in blender_node_tree.nodes:
            outfile.write( "\tnode: {}\n".format( node ) )
            if node.type == 'OUTPUT_MATERIAL':
                self.root_node = node
            for link in blender_node_tree.links:
                if link.from_node == node:
                    outfile.write( "\t\t{}->{}: {}\n".format( link.from_socket.name, link.to_socket.name, link.to_node ) )
                if link.to_node == node:
                    outfile.write( "\t\t{}<-{}: {}\n".format( link.to_socket.name, link.from_socket.name, link.from_node ) )
                    
            

        outfile.write( "**** root node: {}\n".format( self.root_node ) )
            

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
        xml_node.attrib[ "resolution"        ] = "{} {}".format( 
                render.resolution_x, render.resolution_y
                )


    def createDisplay( self, xml_node ):
        xml_node.attrib[ "type" ] = "ImageFileDisplay" 
        filename_param = ET.SubElement( xml_node, "string" )
        filename_param.attrib[ "name"  ] = "filename" 
        filename_param.attrib[ "value" ] = "blender.exr" 


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
        if focal_dist < 1e-6:
            focal_dist = 1.0
        filename_param = ET.SubElement( xml_node, "float" )
        filename_param.attrib[ "name"  ] = "focal_distance" 
        filename_param.attrib[ "value" ] = "{}".format( focal_dist ) 

        h_offset   = math.tan( camera.angle_x*0.5 ) * focal_dist
        v_offset   = math.tan( camera.angle_y*0.5 ) * focal_dist
        lrbt_string = "{} {} {} {}".format( 
                -h_offset, h_offset, -v_offset, v_offset
                )
        view_plane_param = ET.SubElement( xml_node, "vector4" )
        view_plane_param.attrib[ "name"  ] = "view_plane" 
        view_plane_param.attrib[ "value" ] = lrbt_string 

        aperture_param = ET.SubElement( xml_node, "float" )
        aperture_param.attrib[ "name"  ] = "aperture_radius" 
        aperture_param.attrib[ "value" ] = "{}".format( 
                camera.cycles.aperture_size )



    def translateMesh( self, blender_node, xml_node ):
        xml_node.attrib[ "type" ] = "TriMesh" 
        xml_node.attrib[ "name" ] = blender_node.name
        filename_param = ET.SubElement( xml_node, "string" )
        datafile = blender_node.name + ".lmesh"
        filename_param.attrib[ "name"  ] = "datafile" 
        filename_param.attrib[ "value" ] = datafile 

        mesh = blender_node.to_mesh( self.context.scene, True, 'RENDER' )
        mesh.transform( blender_node.matrix_world )
        mesh.calc_normals()
        xml_node.attrib[ "surface" ] = mesh.materials[0].name

        with open( os.path.join( self.dirpath, datafile ), "wb" ) as df:
            verts    = mesh.vertices
            polygons = mesh.polygons
            mats     = mesh.materials
            
            num_tris = 0
            triangles = []
            for poly in mesh.polygons:
                v0 = mesh.loops[poly.loop_start+0].vertex_index
                v1 = mesh.loops[poly.loop_start+1].vertex_index
                v2 = mesh.loops[poly.loop_start+2].vertex_index
                triangles.extend( [ v0, v1, v2 ] )
                num_tris += 1
                for loop_index in range( poly.loop_start+3,
                                         poly.loop_start+poly.loop_total ):
                    v1 = v2
                    v2 = mesh.loops[loop_index].vertex_index
                    triangles.extend( [ v0, v1, v2 ] )
                    num_tris += 1

            num_verts = len(verts)
            vertices = []
            for vert in verts:
                vertices.extend( [ 
                    vert.co[0],     vert.co[1],     vert.co[2],
                    vert.normal[0], vert.normal[1], vert.normal[2] 
                    ] ) 
            
            df.write( struct.pack( 'III', 0x01, num_verts, num_tris ) )
            array.array( 'f', vertices  ).tofile( df )
            array.array( 'I', triangles ).tofile( df )
    

    def translateSurface( self, blender_node, xml_node ):
        material = blender_node
        xml_node.attrib[ "type" ] = "Ward" 
        xml_node.attrib[ "name" ] = blender_node.name

        diff_param = ET.SubElement( xml_node, "color" )
        diff_param.attrib[ "name"  ] = "diffuse_reflectance" 
        diff_param.attrib[ "value" ] = "{:.4} {:.4} {:.4}".format( 
                material.diffuse_color[0] * material.diffuse_intensity,
                material.diffuse_color[1] * material.diffuse_intensity,
                material.diffuse_color[2] * material.diffuse_intensity )

        spec_param = ET.SubElement( xml_node, "color" )
        spec_param.attrib[ "name"  ] = "specular_reflectance" 
        spec_param.attrib[ "value" ] = "{:.4} {:.4} {:.4}".format( 
                material.specular_color[0] * material.specular_intensity,
                material.specular_color[1] * material.specular_intensity,
                material.specular_color[2] * material.specular_intensity )

        alphau_param = ET.SubElement( xml_node, "float" )
        alphau_param.attrib[ "name" ] = "alpha_u"
        alphau_param.attrib[ "value" ] = "{:.4}".format( 
                material.specular_slope )

        alphav_param = ET.SubElement( xml_node, "float" )
        alphav_param.attrib[ "name" ] = "alpha_v"
        alphav_param.attrib[ "value" ] = "{:.4}".format( 
                material.specular_slope )
        
        NodeTree( self.xml_file, blender_node.node_tree )


    def translateLight( self, blender_node, xml_scene ):
        lamp = blender_node.data
        if lamp.type == 'AREA':
            xml_geom_node = ET.SubElement( xml_scene, "geometry" )
            xml_geom_node.attrib[ "type"    ] = "Parallelogram"
            xml_geom_node.attrib[ "surface" ] = "emitter_" + lamp.name
            xml_geom_node.attrib[ "name"    ] = lamp.name

            transform = blender_node.matrix_world
            anchor = mathutils.Vector( ( -lamp.size*0.5, -lamp.size*0.5, 0.0 ) )
            U      = mathutils.Vector( (  lamp.size, 0.0, 0.0 ) )
            V      = mathutils.Vector( (  0.0, lamp.size, 0.0 ) )
            anchor = transform*anchor
            U      = transform.to_3x3()*U
            V      = transform.to_3x3()*V
            color  = lamp.color
        
            anchor_param = ET.SubElement( xml_geom_node, "vector3" )
            anchor_param.attrib[ "name" ] = "anchor"
            anchor_param.attrib[ "value" ] = "{:.4} {:.4} {:.4}".format( 
                    anchor[0], anchor[1], anchor[2] )
            
            U_param = ET.SubElement( xml_geom_node, "vector3" )
            U_param.attrib[ "name" ] = "U"
            U_param.attrib[ "value" ] = "{:.4} {:.4} {:.4}".format( 
                    U[0], U[1], U[2] )
            
            V_param = ET.SubElement( xml_geom_node, "vector3" )
            V_param.attrib[ "name" ] = "V"
            V_param.attrib[ "value" ] = "{:.4} {:.4} {:.4}".format( 
                    V[0], V[1], V[2] )
            
            xml_surf_node = ET.SubElement( xml_scene, "surface" )
            xml_surf_node.attrib[ "type" ] = "DiffuseEmitter"
            xml_surf_node.attrib[ "name" ] = "emitter_" + lamp.name

            radiance_param = ET.SubElement( xml_surf_node, "color" )
            radiance_param.attrib[ "name"  ] = "radiance"
            radiance_param.attrib[ "value" ] = "{:.4} {:.4} {:.4}".format( 
                    color[0], color[1], color[2] )

        pass


    def gatherMaterials( self, mesh_objs ):
        mats = set() 
        for me in mesh_objs:
            mesh = me.data
            for mat in mesh.materials:
                mats.add( mat )

        return list( mats )


    def gatherTextures( self, material_objs, mesh_objs ):
        textures = set() 
        for mat in material_objs:
            for tex_slot in mat.texture_slots:
                if tex_slot:
                    textures.add( tex_slot )
        for mesh_obj in mesh_objs:
            mesh = mesh_obj.data
            for uv_texture in mesh.tessface_uv_textures:
                mesh_texture_face = uv_texture.data
        return list( textures )


    ############################################################################
    #
    # public interface 
    #
    ############################################################################
    def export( self ):

        scene   = self.context.scene
        camera  = scene.camera
        objects     = scene.objects 
        vis_objects = [ obj for obj in scene.objects if obj.is_visible( scene ) ]

        xml_root = ET.Element("legion_scene")
        xml_root.attrib[ "name" ] = "blender"

        xml_camera   = ET.SubElement( xml_root, "camera" )
        self.translateCamera( camera, xml_camera )

        xml_renderer = ET.SubElement( xml_root, "renderer" )
        self.createRenderer( xml_renderer )

        xml_display  = ET.SubElement( xml_root, "display" )
        self.createDisplay( xml_display )

        xml_scene = ET.SubElement( xml_root, "scene" )
        meshes = [ 
                o for o in objects if o.type == 'MESH' and o.is_visible( scene )
                ]
        
        materials = self.gatherMaterials( meshes )
        textures  = self.gatherTextures( materials, meshes )

        for mat in materials:
            xml_mat = ET.SubElement( xml_scene, "surface" )
            self.translateSurface( mat, xml_mat )

        for tex in textures:
            pass

        for mesh in meshes:
            xml_mesh = ET.SubElement( xml_scene, "geometry" )
            self.translateMesh( mesh, xml_mesh )

        lights = [ o for o in objects if o.type == 'LAMP' ]
        for light in lights:
            self.translateLight( light, xml_scene )

        text  = minidom.parseString( ET.tostring( xml_root ) ).toprettyxml()

        self.xml_file.write( text )
        
        return {'FINISHED'}




