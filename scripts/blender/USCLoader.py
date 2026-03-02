# Importer for USC-HairSalon: A 3D Hairstyle Database for Hair Modeling
#  https://www-scf.usc.edu/%7Eliwenhu/SHM/database.html
import bpy
import bmesh
import math
from mathutils import Vector
import struct
import array

data_path = "C:/tmp"      # <------------------- path
hairstyle_id = "00144"    # <------------------- strand id

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
# data of the mesh
vertices = []  # XYZ coords
edges = []
faces = []

fin = open(f"{data_path}/strands{hairstyle_id}.data", "rb")

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
hair_mesh = bpy.data.meshes.new(f"hair_mesh_{hairstyle_id}")
hair_mesh.from_pydata(vertices, edges, faces)
hair_mesh.update()

# create object from mesh
hair_object = bpy.data.objects.new(f"hair style {hairstyle_id}", hair_mesh)
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