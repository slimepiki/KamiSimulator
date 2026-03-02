# Importer for USC-HairSalon: A 3D Hairstyle Database for Hair Modeling
#  https://www-scf.usc.edu/%7Eliwenhu/SHM/database.html
import bpy # type: ignore
import math
from mathutils import Vector# type: ignore
import math
import os
import struct
import array
# import sys
# import os
# from pathlib import Path


def cleanUp():
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def setScene():
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, -3, 1.6), rotation=(math.radians(90), 0, 0), scale=(1, 1, 1))
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(-1, 0.2, 2), rotation=(0, 0, 0), scale=(1, 1, 1))
    # bpy.context.scene.render.film_transparent = True
    # bpy.context.scene.view_settings.view_transform = 'Standard'
    # bpy.context.scene.use_nodes = True
    # bpy.ops.node.add_node(use_transform=True, type="CompositorNodeAlphaOver")
    # bpy.data.scenes["Scene"].node_tree.nodes["Alpha Over"].premul = 1
    # script_dir_path = os.path.dirname(bpy.context.space_data.text.filepath)
    # sys.path.append(script_dir_path)

    # Path_obj_dir = Path(script_dir_path)
    # EXR_Path = f"{Path_obj_dir}/assets/golden_gate_hills_4k.exr"
    # exr_img = bpy.ops.image.open(filepath=EXR_Path, relative_path=True, show_multiview=False)


def addStrand(vertices, edges, strand_data_xyz):
    
    # add first vertex of strand
    xyz_idx = 0 
    vec =  Vector((strand_data_xyz[xyz_idx], strand_data_xyz[xyz_idx+1], strand_data_xyz[xyz_idx+2]))
    vertices.append(vec) 
    
    num_verts_to_add = int(len(strand_data_xyz) / 3)
    edge_vidx = len(vertices)
    
    for i in range(1, num_verts_to_add):
        xyz_idx += 3
        vec =  Vector((strand_data_xyz[xyz_idx], strand_data_xyz[xyz_idx+1], strand_data_xyz[xyz_idx+2]))
        vertices.append(vec) 
        
        edges.append((edge_vidx-1, edge_vidx))
        edge_vidx += 1

# -----------------------------------------
# filename = filenamePrefix + "_hair_" + std::to_string(frameCount) + ".data";

def getUscHair(data_path, hairstyle_id):
    return f"{data_path}/strands{hairstyle_id}.data"

def getHairFramePath(animation_name_prefix, frame_count):
    return f"{animation_name_prefix}_hair_{str(frame_count)}.data"

def getHeadFramePath(animation_name_prefix, frame_count):
    return f"{animation_name_prefix}_head_{str(frame_count)}.data"

def getBodyFramePath(animation_name_prefix, frame_count):
    return f"{animation_name_prefix}_body_{str(frame_count)}.data"

def loadHead(head_path):
     bpy.ops.wm.obj_import(filepath = head_path)
     bpy.context.object.location[1] = 0.01
     bpy.context.object.location[2] = -0.005


def loadHair(hair_path):
    name = f"{os.path.splitext(os.path.basename(hair_path))[0]}"
    vertices = []  # XYZ coords
    edges = []
    faces = []

    fin = open(f"{hair_path}", "rb")

    num_strands = struct.unpack('<i', fin.read(4))[0]
    print("num_strands = ", num_strands)
    assert num_strands == 10000, f"exspected 10000 strands, got: {num_strands}"

    strand_idx = 0
    while (strand_idx < num_strands):

        # read num of verts of strand
        strand_idx = strand_idx + 1    
        num_verts = struct.unpack('<i', fin.read(4))[0]
        assert num_verts == 1 or num_verts == 100, f"num_verts should be 1 or 100, got: {num_verts}"
        
        # read strand
        strand_data_xyz = array.array('f') 
        strand_data_xyz.fromfile(fin, 3 * num_verts) # vert's XYZ corrds

        if (num_verts < 2):  # skip empty roots
            continue
        
        addStrand(vertices, edges, strand_data_xyz.tolist())

    fin.close()

    print("Data read, creating hair object...")

    # create the mesh
    hair_mesh = bpy.data.meshes.new(name)
    hair_mesh.from_pydata(vertices, edges, faces)
    hair_mesh.update()

    # create object from mesh
    hair_object = bpy.data.objects.new(name, hair_mesh)
    # get collection
    collection_name = 'USC-HairSalon (imported hair styles)'
    hair_collection = bpy.data.collections.get(collection_name)
    if hair_collection is None:
        hair_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(hair_collection)
    # add object to scene collection
    hair_collection.objects.link(hair_object)

    # fix rotation (90° X-axis)
    hair_object.rotation_euler[0] = math.radians(90)
    #select object & apply rotation
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = hair_object
    hair_object.select_set(True)
    bpy.ops.object.transform_apply(rotation=True)
    bpy.ops.object.convert(target='CURVE')
    bpy.ops.object.convert(target='CURVES')


    #bpy.ops.mesh.extrude_edges_move(MESH_OT_extrude_edges_indiv={"use_normal_flip":False, "mirror":False}, TRANSFORM_OT_translate={"value":(-0.00949307, 0.0368379, 0.0473253), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
