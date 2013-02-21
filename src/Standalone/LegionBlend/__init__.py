
bl_info = {
        "name":         "Legion Renderer scene format",
        "author":       "R. Keith Morley",
        "blender":      (2,6,5),
        "version":      (0,0,1),
        "location":     "File > Import",
        "description":  "Export Legion data format",
        "category":     "Export"
}


if "bpy" in locals():
    import imp
    if "export_legion" in locals():
        imp.reload(export_legion)


import bpy
from bpy_extras.io_utils import ExportHelper

class ExportLegion(bpy.types.Operator, ExportHelper):
    bl_idname       = "export_scene.lxml";
    bl_label        = "Export Legion";
    bl_options      = {'PRESET'};

    filename_ext    = ".lxml";

    def execute(self, context):
        filepath = self.filepath
        filepath = bpy.path.ensure_ext(filepath, self.filename_ext)
        from . import export_legion
        return export_legion.Exporter( context, filepath ).export()


# Register addon
def menu_func(self, context):
    self.layout.operator( ExportLegion.bl_idname, text="Legion (.lxml)" );

def register():
    bpy.utils.register_module(__name__);
    bpy.types.INFO_MT_file_export.append(menu_func);
    
def unregister():
    bpy.utils.unregister_module(__name__);
    bpy.types.INFO_MT_file_export.remove(menu_func);

if __name__ == "__main__":
    register()
