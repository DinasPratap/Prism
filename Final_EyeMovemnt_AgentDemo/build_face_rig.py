"""
build_face_rig.py  --  generates FaceRig.blend from the MakeHuman CC0 assets in ./assets/makehuman

    blender -b -P build_face_rig.py                  # writes FaceRig.blend next to this file
    blender -b -P build_face_rig.py -- out.blend     # custom output path

WHAT IT BUILDS
  * FaceBody: MakeHuman base mesh (skin faces only, with its UVs), textured with textures/skin_diffuse.jpg
  * FaceRig: the MakeHuman default skeleton (163 bones) with its CC0 skin weights, including bones eye.L / eye.R.
    The eye bones are moved to the eyeball centres and aimed with Damped Track constraints at FIXATION_TARGET.
    (MakeHuman's eye-bone skin weights are dropped so the eyelids stay still and renders are repeatable.)
  * Eyeball_L / Eyeball_R: sclera, glass cornea, procedural iris and a black pupil disc that really changes size.
    Each eyeball is parented to its eye bone.
  * FIXATION_TARGET (35 cm in front of the eye midpoint) and Cam_Locked (40 cm in front, all transforms locked)
  * three-point lighting, a neutral backdrop, Cycles settings for 1024 x 1024 output

COORDINATES: centimetres (1 Blender unit = 1 cm). The eye midpoint is the world origin and the face looks towards -Y.
MakeHuman's character-left eye (eye.L) is at +X, which appears on the image's right.
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
MH = os.path.join(HERE, "assets", "makehuman")
TEX = os.path.join(HERE, "textures", "skin_diffuse.jpg")
OUT = os.path.join(HERE, "FaceRig.blend")
if "--" in sys.argv and len(sys.argv) > sys.argv.index("--") + 1:
    OUT = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

S = 10.0                  # MakeHuman units are decimetres -> centimetres
EYE_R = 1.2               # eyeball radius (cm)
EYE_FORWARD_DM = 0.07     # eyeball centre sits this far in front of MakeHuman's eye joint (fits the socket)
FIX_DIST, CAM_DIST = 35.0, 40.0
GAZE_BONES = ("eye.L", "eye.R")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.scale_length = 0.01
scene.unit_settings.length_unit = "CENTIMETERS"


# ----------------------------------------------------------------------------- 1. parse MakeHuman data
def parse_obj(path):
    verts, uvs, faces, groups, grp = [], [], [], {}, None
    with open(path) as f:
        for ln in f:
            if ln.startswith("v "):
                verts.append(tuple(map(float, ln.split()[1:4])))
            elif ln.startswith("vt "):
                uvs.append(tuple(map(float, ln.split()[1:3])))
            elif ln.startswith("g "):
                grp = ln.split()[1]
                groups.setdefault(grp, set())
            elif ln.startswith("f "):
                corners = []
                for tok in ln.split()[1:]:
                    p = tok.split("/")
                    corners.append((int(p[0]) - 1, int(p[1]) - 1 if len(p) > 1 and p[1] else -1))
                groups[grp].update(c[0] for c in corners)
                faces.append((grp, corners))
    return verts, uvs, faces, groups


mh_verts, mh_uvs, mh_faces, mh_groups = parse_obj(os.path.join(MH, "base.obj"))
skeleton = json.load(open(os.path.join(MH, "default.mhskel")))
weights = json.load(open(os.path.join(MH, "default_weights.mhw")))["weights"]


def centroid(idx):
    pts = [mh_verts[i] for i in idx]
    return tuple(sum(c) / len(pts) for c in zip(*pts))


eye_mh = {}
for side, grp in (("L", "helper-l-eye"), ("R", "helper-r-eye")):
    c = centroid(sorted(mh_groups[grp]))
    eye_mh[side] = (c[0], c[1], c[2] + EYE_FORWARD_DM)
mid_mh = tuple((a + b) / 2 for a, b in zip(eye_mh["L"], eye_mh["R"]))


def T(p):
    """MakeHuman (x right-of-character, y up, z forward, dm)  ->  Blender cm with the eye midpoint at the origin, face to -Y."""
    return Vector(((p[0] - mid_mh[0]) * S, -(p[2] - mid_mh[2]) * S, (p[1] - mid_mh[1]) * S))


# ----------------------------------------------------------------------------- 2. materials
def setin(node, names, value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return


def principled(name, color, rough=0.5, sss=None, spec=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    setin(b, ["Base Color"], (*color, 1))
    setin(b, ["Roughness"], rough)
    if sss is not None:
        setin(b, ["Subsurface Weight", "Subsurface"], sss)
        setin(b, ["Subsurface Radius"], (1.0, 0.35, 0.2))
    if spec is not None:
        setin(b, ["Specular IOR Level", "Specular"], spec)
    return mat


def mix_node(nt):
    try:
        n = nt.nodes.new("ShaderNodeMixRGB")
        n.blend_type = "MULTIPLY"
        return n, n.inputs[0], n.inputs[1], n.inputs[2], n.outputs[0]
    except Exception:
        n = nt.nodes.new("ShaderNodeMix")
        n.data_type, n.blend_type = "RGBA", "MULTIPLY"
        return n, n.inputs[0], n.inputs[6], n.inputs[7], n.outputs[2]


def skin_material():
    mat = principled("Skin", (0.8, 0.6, 0.5), 0.5, sss=0.06, spec=0.4)
    nt = mat.node_tree
    img = bpy.data.images.load(TEX)
    img.pack()                                   # texture travels inside the .blend
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = 900, 4
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    return mat


def iris_material(radius=0.58):
    mat = principled("Iris", (0.3, 0.17, 0.06), 0.3, spec=0.6)
    nt = mat.node_tree
    N, L = nt.nodes, nt.links.new
    coord, sep, comb, vlen = N.new("ShaderNodeTexCoord"), N.new("ShaderNodeSeparateXYZ"), N.new("ShaderNodeCombineXYZ"), N.new("ShaderNodeVectorMath")
    vlen.operation = "LENGTH"
    rmap = N.new("ShaderNodeMapRange")
    rmap.inputs["From Max"].default_value = radius
    ramp = N.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.0, (0.10, 0.04, 0.015, 1)
    for pos, col in ((0.22, (0.62, 0.34, 0.09, 1)), (0.6, (0.32, 0.18, 0.06, 1)), (0.88, (0.14, 0.09, 0.05, 1))):
        cr.elements.new(pos).color = col
    cr.elements[-1].position, cr.elements[-1].color = 1.0, (0.015, 0.01, 0.008, 1)
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = 38, 6
    nramp = N.new("ShaderNodeValToRGB")
    nramp.color_ramp.elements[0].position, nramp.color_ramp.elements[1].position = 0.3, 0.75
    nramp.color_ramp.elements[0].color, nramp.color_ramp.elements[1].color = (0.55, 0.5, 0.45, 1), (1, 1, 1, 1)
    _m, fac, a, b, out = mix_node(nt)
    fac.default_value = 0.7
    L(coord.outputs["Object"], sep.inputs["Vector"])
    L(sep.outputs["X"], comb.inputs["X"])
    L(sep.outputs["Z"], comb.inputs["Z"])
    L(comb.outputs["Vector"], vlen.inputs[0])
    L(vlen.outputs["Value"], rmap.inputs["Value"])
    L(rmap.outputs["Result"], ramp.inputs["Fac"])
    L(coord.outputs["Object"], noise.inputs["Vector"])
    L(noise.outputs["Fac"], nramp.inputs["Fac"])
    L(ramp.outputs["Color"], a)
    L(nramp.outputs["Color"], b)
    L(out, N["Principled BSDF"].inputs["Base Color"])
    return mat


sclera_mat = principled("Sclera", (0.93, 0.89, 0.85), 0.32, sss=0.05, spec=0.5)
cornea_mat = principled("Cornea", (1, 1, 1), 0.0)
setin(cornea_mat.node_tree.nodes["Principled BSDF"], ["Transmission Weight", "Transmission"], 1.0)
setin(cornea_mat.node_tree.nodes["Principled BSDF"], ["IOR"], 1.376)
iris_mat, pupil_mat = iris_material(), principled("Pupil", (0, 0, 0), 0.2)

# ----------------------------------------------------------------------------- 3. body mesh (skin faces only) + weights
bone_names = list(skeleton["bones"].keys())
body_mesh = bpy.data.meshes.new("FaceBody")
body = bpy.data.objects.new("FaceBody", body_mesh)
scene.collection.objects.link(body)
for bn in bone_names:
    body.vertex_groups.new(name=bn)

bm = bmesh.new()
deform = bm.verts.layers.deform.verify()      # layers must exist before elements are created
uv_layer = bm.loops.layers.uv.new("UVMap")
bverts = [bm.verts.new(T(p)) for p in mh_verts]
bm.verts.ensure_lookup_table()
for grp, corners in mh_faces:
    if grp != "body":
        continue
    try:
        face = bm.faces.new([bverts[c[0]] for c in corners])
    except ValueError:
        continue                               # duplicate / degenerate face
    for loop, (_vi, ti) in zip(face.loops, corners):
        if ti >= 0:
            loop[uv_layer].uv = mh_uvs[ti]
for bone, pairs in weights.items():
    if bone in GAZE_BONES or bone not in body.vertex_groups:
        continue                               # keep the eyelids still: no eye-bone skin weights
    gi = body.vertex_groups[bone].index
    for vi, w in pairs:
        bverts[vi][deform][gi] = w
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
bm.to_mesh(body_mesh)
bm.free()
body_mesh.polygons.foreach_set("use_smooth", [True] * len(body_mesh.polygons))
body_mesh.materials.append(skin_material())

# ----------------------------------------------------------------------------- 4. armature
arm = bpy.data.armatures.new("FaceRigData")
arm.display_type = "STICK"
rig = bpy.data.objects.new("FaceRig", arm)
rig.show_in_front = True
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
joint = lambda name: T(centroid(skeleton["joints"][name]))
for bn in bone_names:
    spec = skeleton["bones"][bn]
    eb = arm.edit_bones.new(bn)
    eb.head, eb.tail = joint(spec["head"]), joint(spec["tail"])
    if (eb.tail - eb.head).length < 1e-4:
        eb.tail = eb.head + Vector((0, 0, 0.1))
for bn in bone_names:
    parent = skeleton["bones"][bn]["parent"]
    if parent:
        arm.edit_bones[bn].parent = arm.edit_bones[parent]
eye_center = {s: T(eye_mh[s]) for s in "LR"}
for s in "LR":                                 # eye bones: pivot at the eyeball centre, pointing forward (-Y)
    eb = arm.edit_bones[f"eye.{s}"]
    eb.head, eb.tail = eye_center[s], eye_center[s] + Vector((0, -1.5, 0))
    eb.roll = 0.0
bpy.ops.object.mode_set(mode="OBJECT")
mod = body.modifiers.new("Armature", "ARMATURE")
mod.object = rig
body.parent = rig

# ----------------------------------------------------------------------------- 5. eyeballs, fixation target, camera
def finish(name, bm_, mat, parent=None):
    mesh = bpy.data.meshes.new(name)
    bm_.to_mesh(mesh)
    bm_.free()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    if parent:
        obj.parent = obj_parent = parent
    return obj


def sphere(radius, u, v):
    bm_ = bmesh.new()
    bmesh.ops.create_uvsphere(bm_, u_segments=u, v_segments=v, radius=radius)
    return bm_


def disc(radius):
    bm_ = bmesh.new()
    bmesh.ops.create_circle(bm_, cap_ends=True, cap_tris=True, segments=64, radius=radius)
    bmesh.ops.rotate(bm_, verts=bm_.verts[:], cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "X"))  # face -Y
    return bm_


def bone_parent(obj, bone):
    """Parent obj to a bone while keeping its current world transform (independent of Blender's tail-offset convention)."""
    target_world = obj.matrix_world.copy()
    obj.matrix_basis = Matrix.Identity(4)
    obj.parent, obj.parent_type, obj.parent_bone = rig, "BONE", bone
    obj.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()
    parent_effective = obj.matrix_world.copy()      # with identity basis + inverse, world == the bone's effective matrix
    obj.matrix_parent_inverse = parent_effective.inverted() @ target_world


target = bpy.data.objects.new("FIXATION_TARGET", None)
target.empty_display_type, target.empty_display_size = "SPHERE", 0.8
target.location = (0, -FIX_DIST, 0)
scene.collection.objects.link(target)

for s in "LR":
    eye = finish(f"Eyeball_{s}", sphere(EYE_R, 64, 40), sclera_mat)
    eye.location = eye_center[s]
    cornea = finish(f"Cornea_{s}", sphere(0.78, 48, 32), cornea_mat, parent=eye)
    cornea.location = (0, -0.67, 0)
    cornea.visible_shadow = False             # lets light reach the iris through the glass (no caustic noise)
    iris = finish(f"Iris_{s}", disc(0.58), iris_mat, parent=eye)
    iris.location = (0, -1.10, 0)
    pupil = finish(f"Pupil_{s}", disc(1.0), pupil_mat, parent=eye)
    pupil.location = (0, -1.112, 0)
    pupil.scale = (0.2, 1.0, 0.2)              # 4 mm diameter
    bpy.context.view_layer.update()
    bone_parent(eye, f"eye.{s}")
    pb = rig.pose.bones[f"eye.{s}"]
    c = pb.constraints.new("DAMPED_TRACK")
    c.target, c.track_axis = target, "TRACK_Y"

cam_data = bpy.data.cameras.new("Cam_Locked")
cam_data.lens, cam_data.sensor_width, cam_data.sensor_fit = 85, 36, "HORIZONTAL"
cam = bpy.data.objects.new("Cam_Locked", cam_data)
cam.location, cam.rotation_euler = (0, -CAM_DIST, 0), (math.pi / 2, 0, 0)
cam.lock_location = cam.lock_rotation = cam.lock_scale = (True, True, True)
scene.collection.objects.link(cam)
scene.camera = cam

# ----------------------------------------------------------------------------- 6. lights, world, render defaults
def area_light(name, loc, energy, size):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size = energy, size
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    scene.collection.objects.link(lo)
    tc = lo.constraints.new("TRACK_TO")
    tc.target, tc.track_axis, tc.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"


area_light("Key", (-45, -55, 40), 30000, 14)       # small lights keep corneal reflections tiny
area_light("Fill", (50, -55, 5), 12000, 14)
area_light("Top", (0, -30, 60), 9000, 12)
world = bpy.data.worlds.new("Backdrop")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.30, 0.31, 0.33, 1)
scene.world = world
r = scene.render
r.resolution_x = r.resolution_y = 1024
r.resolution_percentage = 100
r.engine = "CYCLES"
scene.cycles.samples, scene.cycles.seed, scene.cycles.use_animated_seed = 48, 0, False
scene.view_settings.view_transform = "Filmic" if "Filmic" in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items else "Standard"
scene["fixation_distance_cm"], scene["camera_distance_cm"] = FIX_DIST, CAM_DIST

bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
print("Saved", OUT)
