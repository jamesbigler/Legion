
import bpy
import xml.etree.ElementTree as ET

'''
class Node:
    def __init__( self, blender_node ):
    self.blender_node = blender_node
    self.inputs       = [] # NodeLinks  
'''



class Node:

    def __init__( self, material_name, blender_node, is_material_root=False ):
        self.blender_node = blender_node
        self.inputs       = [] # NodeLinks  
        self.name         = material_name if is_material_root else material_name+"."+blender_node.name


    def legionPluginCategory( self ):
        raise NotImplementedError("legionPluginCategory abstract method")
    

    def legionPluginType( self ):
        raise NotImplementedError("legionPluginType abstract method")


    def mapInput( self, idx, name ):
        raise NotImplementedError("mapInputs abstract method")

    
    @staticmethod
    def socketDefaultValue( socket ):
        v = socket.default_value
        if( socket.type == 'VECTOR' ):
            return ('vector3', '{:.4} {:.4} {:.4}'.format( v[0], v[1], v[2] ) )
        elif( socket.type == 'RGBA' ):
            return ('color', '{:.4} {:.4} {:.4}'.format( v[0], v[1], v[2] ) )
        elif( socket.type == 'VALUE' ):
            return ('float', '{:.4}'.format( v ) )
        else:
            raise RuntimeError("Unhandled socket type {}:{}'".format(
                socket.name, socket.type ) )
        

    def toXML( self, xml_scene, already_visited, override_name = "" ):
        if self in already_visited:
            return
        already_visited.add( self )

        # Depth first so that child surfaces are created first in XML
        for idx, (socket, input_link, input_node) in enumerate( self.inputs ):
            mapped_input_name = self.mapInput( idx, socket.name )
            if not mapped_input_name:
                continue
            if input_link:
                input_node.toXML( xml_scene, already_visited )

        # Now that we have created referenced surfaces ...
        xml_node = ET.SubElement( xml_scene, self.legionPluginCategory() )
        xml_node.attrib[ "name" ] = override_name if override_name else self.name 
        xml_node.attrib[ "type" ] = self.legionPluginType() 

        for idx, (socket, input_link, input_node) in enumerate( self.inputs ):
            mapped_input_name = self.mapInput( idx, socket.name )
            if not mapped_input_name:
                continue
            if input_link:
                input_xml_node = ET.SubElement( xml_node,
                        input_node.legionPluginCategory() )
                input_xml_node.attrib[ "name"  ] = mapped_input_name 
                input_xml_node.attrib[ "value" ] = input_node.name 

            else:
                input_xml_node = ET.SubElement( xml_node, "texture" )
                type, value = Node.socketDefaultValue( socket )
                input_xml_node.attrib[ "name"  ] = mapped_input_name 
                input_xml_node.attrib[ "type"  ] = type
                input_xml_node.attrib[ "value" ] = value 


class TextureNode( Node ):
    def legionPluginCategory( self ):
        return "texture"
    

class SurfaceNode( Node ):
    def legionPluginCategory( self ):
        return "surface"


class MIX_SHADER( SurfaceNode ):
    def legionPluginType( self ):
        return "Mixture" 
    
    def mapInput( self, idx, name ):
        return [ 'mixture_weight', 's0', 's1' ][ idx ]


class BSDF_GLOSSY( SurfaceNode ):
    def legionPluginType( self ):
        return "Beckmann" 
    
    def mapInput( self, idx, name ):
        return [ 'reflectance', 'alpha', '' ][ idx ]


class BSDF_DIFFUSE( SurfaceNode ):
    def legionPluginType( self ):
        return "Lambertian" 
    
    def mapInput( self, idx, name ):
        return [ 'reflectance', '', '' ][ idx ]


class BSDF_GLASS( SurfaceNode ):
    def legionPluginType( self ):
        return "Dielectric" 
    
    def mapInput( self, idx, name ):
        print( 'glass param {}'.format( name ) )
        return [ 'transmitance', '', 'ior_in', ''][ idx ]


class BSDF_TRANSLUCENT( SurfaceNode ):
    def legionPluginType( self ):
        return "Dielectric" 
    
    def mapInput( self, idx, name ):
        print( 'translucent param {}'.format( name ) )
        return [ '', '', '', '', '', '', '', ''][ idx ]


class EMISSION( SurfaceNode ):
    def legionPluginType( self ):
        return "DiffuseEmitter" 
    
    def mapInput( self, idx, name ):
        print( 'emission param {}'.format( name ) )
        return [ '', '', '', '', '', '', '', ''][ idx ]


class TEX_NOISE( TextureNode ):
    def legionPluginType( self ):
        return "PerlinTexture" 
    
    def mapInput( self, idx, name ):
        print( 'texnoise param {}'.format( name ) )
        return [ '', '', '', '', '', '', '', ''][ idx ]


class TEX_IMAGE( TextureNode ):
    def legionPluginType( self ):
        return "ImageTexture" 
    
    def mapInput( self, idx, name ):
        print( 'imagetex param {}'.format( name ) )
        return [ '', '', '', '', '', '', '', ''][ idx ]


class FRESNEL( TextureNode ):
    def legionPluginType( self ):
        return "FresnelSchlickTexture" 
    
    def mapInput( self, idx, name ):
        print( 'fresneltex param {}'.format( name ) )
        return [ '', '', '', '', '', '', '', ''][ idx ]


class MATH( TextureNode ):
    def legionPluginType( self ):
        return "MathTexture" 
    
    def mapInput( self, idx, name ):
        print( 'mathtex param {}'.format( name ) )
        return [ '', '', '', '', '', '', '', ''][ idx ]


class TEX_CHECKER( TextureNode ):
    def legionPluginType( self ):
        return "CheckerTexture" 
    
    def mapInput( self, idx, name ):
        return [ '', 'c0', 'c1', 'scale' ][ idx ]


class RGB( TextureNode ):
    def legionPluginType( self ):
        return "ConstantTexture" 
    
    def mapInput( self, idx, name ):
        print( 'mathtex param {}'.format( name ) )
        return [ '', '', '', '', '', '', '', ''][ idx ]



class OUTPUT_MATERIAL( Node ):
    pass


class NodeTree:

    surfaces_processed = set([]) 


    @staticmethod
    def alreadyProcessed( blender_node_tree ):
        return blender_node_tree in NodeTree.surfaces_processed

    
    def __init__( self, name, blender_node_tree):
        self.node_lookup = {} 
        self.name = name
        NodeTree.surfaces_processed.add( blender_node_tree )

        # Create all nodes and place in lookup dict
        for blender_node in blender_node_tree.nodes:
            node = eval( blender_node.type + "( name, blender_node )" )
            self.node_lookup[ blender_node ] = node
            if blender_node.type == 'OUTPUT_MATERIAL':
                self.root_node = node 

        # Gather all inputs for this node
        for blender_node in blender_node_tree.nodes:
            node = self.node_lookup[ blender_node ]
            for socket in blender_node.inputs:
                if socket.is_linked:
                    for link in blender_node_tree.links:
                        if link.to_socket == socket:
                            from_node = self.node_lookup[ link.from_node ]
                            node.inputs.append( ( socket, link, from_node ) )
                else:
                    node.inputs.append( ( socket, None, None) )

        self.nodes = self.node_lookup.values()


    def toSurfaceXML(  self, xml_scene ):
        # Find the Surface root node
        for socket, input_link, input_node in self.root_node.inputs:
            if input_link and input_link.to_socket.name == "Surface":
                input_node.toXML( xml_scene, set( [] ), self.name )



def translate( material_name, material_node_tree, xml_scene ):

    if NodeTree.alreadyProcessed( material_node_tree ):
        return

    node_tree = NodeTree( material_name, material_node_tree )
    node_tree.toSurfaceXML( xml_scene )
