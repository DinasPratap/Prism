"""
build_artificial_eye.py  --  generates ArtificialEye.blend

    blender -b -P build_artificial_eye.py                      # writes ArtificialEye.blend next to this file
    blender -b -P build_artificial_eye.py -- out.blend         # custom output path

WHAT IT BUILDS  (units: 1 Blender unit = 1 cm; eyeball diameter 2.4 cm; eye spacing 6.3 cm)
  * two eyeballs (sclera, glass cornea, procedural iris, pupil that dilates)
  * the six extrinsic muscles per eye: medial / lateral / superior / inferior rectus, superior / inferior oblique
      - the muscle belly is skinned to a bone with a Stretch To constraint, so it really lengthens, shortens and
        bulges as the globe rotates; the tendon runs along the globe surface to its insertion
      - the superior oblique passes through a trochlea (pulley)
      - muscle colour is driven by a schematic activation value (pale = relaxed, bright red = active)
  * two optic nerves, upper and lower eyelids (blink + lid-follow-gaze)
  * EYE_CONTROL empty: the ONE place that moves everything (gaze_x_deg, gaze_y_deg, vergence_deg, blink,
    pupil_mm, show_lids), all through native drivers, so no script needs to run for the rig to work
  * artificial_eye_api.py embedded as a text block (sidebar panel, script runner, eye_twin loaders, live link)
  * 3 cameras, studio lights, and a 9 second demo animation

COORDINATES: eye faces -Y. Looking at the face from the front camera (at -Y), image-right is +X.
  gaze_x_deg > 0 turns the eyes to +X (image right); gaze_y_deg > 0 looks up.
  Subject's RIGHT eye is at x = -3.15 (left of the image); its nasal side is +X.
"""
import importlib.util
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ArtificialEye.blend")
if "--" in sys.argv and len(sys.argv) > sys.argv.index("--") + 1:
    OUT = os.path.abspath(sys.argv[sys.argv.index("--") + 1])

R = 1.2            # globe radius (cm)
EYE_X = 3.15       # half the interpupillary distance
APEX = 4.8         # orbital apex (muscle origins) is this far behind the globe centre
D2R = math.pi / 180.0
K = "0.0174533"    # degrees -> radians, written as a constant so drivers stay simple expressions

# ----------------------------------------------------------------------------- setup
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.scale_length = 0.01          # 1 BU = 1 cm
scene.unit_settings.length_unit = "CENTIMETERS"
scene.render.fps = 30
scene.render.resolution_x, scene.render.resolution_y = 1200, 800


def collection(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c


C_EYES, C_MUSC, C_LIDS, C_RIG, C_STUDIO = (collection(n) for n in ("Eyes", "Muscles", "Eyelids", "Rig", "Studio"))


# ----------------------------------------------------------------------------- materials
def setin(node, names, value):
    """Set the first input that exists (Blender 3.x and 4.x name some sockets differently)."""
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def principled(name, color, rough=0.5, **kw):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    setin(b, ["Base Color"], (*color, 1))
    setin(b, ["Roughness"], rough)
    if "sss" in kw:
        setin(b, ["Subsurface Weight", "Subsurface"], kw["sss"])
        setin(b, ["Subsurface Radius"], (1.0, 0.35, 0.2))
    if "spec" in kw:
        setin(b, ["Specular IOR Level", "Specular"], kw["spec"])
    return mat


def mix_node(nt, name):
    try:
        n = nt.nodes.new("ShaderNodeMixRGB")
        n.blend_type = "MIX"
        sockets = (n.inputs[0], n.inputs[1], n.inputs[2], n.outputs[0])
    except Exception:
        n = nt.nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        sockets = (n.inputs[0], n.inputs[6], n.inputs[7], n.outputs[2])
    n.name = name
    return (n, *sockets)


def muscle_material(name):
    """Red muscle that brightens with activation. The 'Activation' mix factor gets a driver later."""
    mat = principled(name, (0.55, 0.2, 0.18), 0.45, sss=0.1, spec=0.5)
    nt = mat.node_tree
    _n, fac, a, b, out = mix_node(nt, "Activation")
    a.default_value = (0.50, 0.20, 0.19, 1)     # relaxed
    b.default_value = (1.0, 0.10, 0.04, 1)      # active
    fac.default_value = 0.15
    nt.links.new(out, nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def sclera_material():
    mat = principled("Sclera", (0.93, 0.89, 0.85), 0.32, sss=0.08, spec=0.5)
    nt = mat.node_tree
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.82, 0.74, 0.70, 1)
    ramp.color_ramp.elements[1].color = (0.95, 0.92, 0.88, 1)
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.35, 0.65
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def iris_material(radius):
    mat = principled("Iris", (0.3, 0.17, 0.06), 0.3, spec=0.6)
    nt = mat.node_tree
    N = nt.nodes
    coord, sep, comb, vlen = N.new("ShaderNodeTexCoord"), N.new("ShaderNodeSeparateXYZ"), N.new("ShaderNodeCombineXYZ"), N.new("ShaderNodeVectorMath")
    vlen.operation = "LENGTH"
    rmap = N.new("ShaderNodeMapRange")
    rmap.inputs["From Max"].default_value = radius
    ramp = N.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position, cr.elements[0].color = 0.0, (0.10, 0.04, 0.015, 1)
    for pos, col in ((0.22, (0.62, 0.34, 0.09, 1)), (0.6, (0.32, 0.18, 0.06, 1)), (0.88, (0.14, 0.09, 0.05, 1))):
        e = cr.elements.new(pos)
        e.color = col
    cr.elements[-1].position, cr.elements[-1].color = 1.0, (0.015, 0.01, 0.008, 1)
    noise = N.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = 38, 6
    nramp = N.new("ShaderNodeValToRGB")
    nramp.color_ramp.elements[0].position, nramp.color_ramp.elements[1].position = 0.3, 0.75
    nramp.color_ramp.elements[0].color = (0.55, 0.5, 0.45, 1)
    nramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    mixn, fac, a, b, out = mix_node(nt, "FibreMix")
    try:
        mixn.blend_type = "MULTIPLY"
    except Exception:
        pass
    fac.default_value = 0.7
    L = nt.links.new
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


def cornea_material():
    mat = principled("Cornea", (1, 1, 1), 0.0)
    b = mat.node_tree.nodes["Principled BSDF"]
    setin(b, ["Transmission Weight", "Transmission"], 1.0)
    setin(b, ["IOR"], 1.376)
    return mat


MAT = {
    "sclera": sclera_material(),
    "iris": iris_material(0.58),
    "pupil": principled("Pupil", (0.0, 0.0, 0.0), 0.2),
    "cornea": cornea_material(),
    "tendon": principled("Tendon", (0.92, 0.88, 0.80), 0.4, sss=0.05),
    "nerve": principled("OpticNerve", (0.88, 0.80, 0.55), 0.5, sss=0.05),
    "skin": principled("EyelidSkin", (0.80, 0.54, 0.44), 0.55, sss=0.2),
    "metal": principled("Trochlea", (0.6, 0.6, 0.65), 0.3),
}


# ----------------------------------------------------------------------------- mesh helpers
def finish(name, bm, coll, mat=None, parent=None, smooth=True):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if mat:
        mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    if parent:
        obj.parent = parent
    return obj


def sphere_bm(radius, u=48, v=32, keep=None):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=radius)
    if keep:
        bmesh.ops.delete(bm, geom=[vt for vt in bm.verts if not keep(vt.co)], context="VERTS")
    return bm


def tube_bm(pts, radii, sides=14, cap=True):
    """A tube along a polyline with per-point radius (parallel-transport frames)."""
    bm = bmesh.new()
    n = len(pts)
    tans = [(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized() for i in range(n)]
    ref = Vector((0, 0, 1)) if abs(tans[0].z) < 0.9 else Vector((1, 0, 0))
    nrm = tans[0].cross(ref).normalized()
    rings = []
    for i, (p, t, r) in enumerate(zip(pts, tans, radii)):
        nrm = (nrm - nrm.dot(t) * t).normalized()
        bnm = t.cross(nrm)
        rings.append([bm.verts.new(p + (math.cos(a) * nrm + math.sin(a) * bnm) * r)
                      for a in (2 * math.pi * k / sides for k in range(sides))])
    for i in range(n - 1):
        for k in range(sides):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % sides], rings[i + 1][(k + 1) % sides], rings[i + 1][k]))
    if cap:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def line_pts(a, b, n):
    return [a.lerp(b, i / (n - 1)) for i in range(n)]


def belly_radii(n, rmax):
    return [rmax * (0.35 + 0.65 * max(0.0, math.sin(math.pi * i / (n - 1))) ** 0.7) for i in range(n)]


def empty(name, loc, coll, parent=None, size=0.08, kind="SPHERE"):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type, e.empty_display_size = kind, size
    e.location = loc
    coll.objects.link(e)
    if parent:
        e.parent = parent
    return e


# ----------------------------------------------------------------------------- control empty + drivers
ctrl = empty(
    "EYE_CONTROL", (0, -9, 5.5), C_RIG, size=1.0, kind="ARROWS")
PROPS = {  # name: (default, min, max, description)
    "gaze_x_deg": (0.0, -50.0, 50.0, "Horizontal gaze in degrees (+ = right of the front-camera image)"),
    "gaze_y_deg": (0.0, -40.0, 40.0, "Vertical gaze in degrees (+ = up)"),
    "vergence_deg": (0.0, 0.0, 40.0, "Total convergence of the two eyes in degrees"),
    "blink": (0.0, 0.0, 1.0, "Eyelid closure: 0 open, 1 closed"),
    "pupil_mm": (4.0, 1.5, 9.0, "Pupil diameter in millimetres"),
    "show_lids": (1.0, 0.0, 1.0, "1 shows the eyelids, 0 hides them so the muscles are visible"),
}
for name, (dflt, lo, hi, desc) in PROPS.items():
    ctrl[name] = dflt
    ctrl.id_properties_ui(name).update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=desc)


def drive(idblock, path, expr, names, index=-1):
    """Add a native driver. `names` maps expression variable -> property name on EYE_CONTROL."""
    fc = idblock.driver_add(path, index) if index >= 0 else idblock.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    for var, prop in names.items():
        v = d.variables.new()
        v.name, v.type = var, "SINGLE_PROP"
        v.targets[0].id = ctrl
        v.targets[0].data_path = f'["{prop}"]'
    return fc


GX, GY, VG, BL, PU, SL = (
    {"gx": "gaze_x_deg"}, {"gy": "gaze_y_deg"}, {"v": "vergence_deg"}, {"b": "blink"}, {"p": "pupil_mm"}, {"s": "show_lids"})


def act_expr(muscle, side):
    """Schematic activation 0.15 .. 1.0 (same formulas as ArtificialEye.activations)."""
    yaw = f"(gx+v*0.5)" if side == "R" else f"(gx-v*0.5)"
    add = yaw if side == "R" else f"-{yaw}"            # positive = adduction
    c = lambda x: f"max(0,min(1,{x}))"
    return {
        "MR": f"0.15+0.85*{c(f'({add})/40')}",
        "LR": f"0.15+0.85*{c(f'-({add})/40')}",
        "SR": f"0.15+0.85*{c('gy/35')}",
        "IR": f"0.15+0.85*{c('-gy/35')}",
        "SO": f"0.15+0.51*{c('-gy/35')}",
        "IO": f"0.15+0.51*{c('gy/35')}",
    }[muscle]


# ----------------------------------------------------------------------------- armature (muscle bones)
arm = bpy.data.armatures.new("EyeRigData")
arm.display_type = "STICK"
rig = bpy.data.objects.new("EyeRig", arm)
rig.show_in_front = True
C_RIG.objects.link(rig)
bone_specs = []   # (name, head, tail)


def pt_on_globe(s, phi_deg, dirn, radius):
    """Point at angle phi from the front pole (-Y), heading in direction dirn = (nasal, _, up)."""
    p = math.radians(phi_deg)
    return radius * (math.cos(p) * Vector((0, -1, 0)) + math.sin(p) * Vector((s * dirn[0], 0, dirn[2])))


# name: origin (nasal, y, up), wrap angle/dir, insertion angle/dir, belly radius. Obliques: see below.
MUSCLES = {
    "MR": dict(O=(0.35, APEX, 0.0), wrap=(88, (1, 0, 0)), ins=(57, (1, 0, 0)), r=0.22),
    "LR": dict(O=(-0.35, APEX, 0.0), wrap=(88, (-1, 0, 0)), ins=(64, (-1, 0, 0)), r=0.22),
    "SR": dict(O=(0.0, APEX, 0.35), wrap=(88, (0, 0, 1)), ins=(68, (0, 0, 1)), r=0.20),
    "IR": dict(O=(0.0, APEX, -0.35), wrap=(88, (0, 0, -1)), ins=(62, (0, 0, -1)), r=0.20),
    "SO": dict(O=(0.55, APEX, 0.55), T=(1.15, -0.2, 1.7), wrap=(100, (-0.3, 0, 0.954)), ins=(125, (-0.6, 0, 0.8)), r=0.17),
    "IO": dict(O=(1.0, -0.9, -1.7), wrap=(100, (-0.3, 0, -0.954)), ins=(125, (-0.6, 0, -0.8)), r=0.17),
}
WRAP_R = R + 0.055   # tendons/wrap points sit just above the sclera

built = {}           # side -> dict of objects
pending = []         # (bone name, target empty, rest length, volume mode)
skinned = []         # (mesh object, bone name)

for side, s in (("R", 1), ("L", -1)):
    cx = -s * EYE_X
    Cc = Vector((cx, 0, 0))
    sclera = finish(f"Eyeball_{side}", sphere_bm(R, 64, 40), C_EYES, MAT["sclera"])
    sclera.location = Cc
    eye = sclera
    cornea = finish(f"Cornea_{side}", sphere_bm(0.78, 48, 32), C_EYES, MAT["cornea"], parent=eye)
    cornea.location = (0, -0.67, 0)
    # iris and pupil: discs in the XZ plane facing -Y
    for nm, rad, y, mk in (("Iris", 0.58, -1.10, "iris"), ("Pupil", 1.0, -1.112, "pupil")):
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, cap_tris=True, segments=64, radius=rad)
        bmesh.ops.rotate(bm, verts=bm.verts[:], cent=(0, 0, 0), matrix=__import__("mathutils").Matrix.Rotation(math.pi / 2, 3, "X"))
        o = finish(f"{nm}_{side}", bm, C_EYES, MAT[mk], parent=eye)
        o.location = (0, y, 0)
        built.setdefault(side, {})[nm] = o
    pupil = built[side]["Pupil"]
    drive(pupil, "scale", "p/20", PU, 0)       # radius (cm) = diameter (mm) / 20
    drive(pupil, "scale", "p/20", PU, 2)

    # eyeball rotation: Rz(yaw) . Rx(-pitch)  (XYZ euler). Vergence turns each eye towards the nose.
    yaw = f"(gx+v*0.5)*{K}" if side == "R" else f"(gx-v*0.5)*{K}"
    drive(eye, "rotation_euler", f"-gy*{K}", GY, 0)
    drive(eye, "rotation_euler", yaw, {**GX, **VG}, 2)

    # --- muscles
    nerve_target = empty(f"NervePole_{side}", Vector((0.26 * s, 0.966, 0)) * (R - 0.01), C_RIG, parent=eye, size=0.05)
    for mname, m in MUSCLES.items():
        O = Vector((s * m["O"][0], m["O"][1], m["O"][2]))
        W = pt_on_globe(s, m["wrap"][0], m["wrap"][1], WRAP_R)
        P = pt_on_globe(s, m["ins"][0], m["ins"][1], WRAP_R)
        wemp = empty(f"Wrap_{mname}_{side}", W, C_RIG, parent=eye, size=0.07)
        # tendon strip along the globe surface from the wrap point to the insertion
        u0, u1 = W.normalized(), P.normalized()
        arc = [u0.slerp(u1, i / 9) * WRAP_R for i in range(10)]
        tend = finish(f"Tendon_{mname}_{side}", tube_bm(arc, [0.072 - 0.02 * i / 9 for i in range(10)], 10), C_MUSC, MAT["tendon"], parent=eye)
        # contractile belly (or, for the superior oblique, a tendon through the trochlea)
        if mname == "SO":
            T = Vector((s * m["T"][0], m["T"][1], m["T"][2]))
            belly = finish(f"Belly_SO_{side}", tube_bm(line_pts(Cc + O, Cc + T, 12), belly_radii(12, m["r"])), C_MUSC, None)
            belly.data.materials.append(muscle_material(f"Muscle_SO_{side}"))
            troch = finish(f"Trochlea_{side}", sphere_bm(0.13, 16, 10), C_MUSC, MAT["metal"])
            troch.location = Cc + T
            head, tail = Cc + T, Cc + W
            mesh_obj = finish(f"Bone_SO_{side}", tube_bm(line_pts(head, tail, 12), [0.06] * 12, 10), C_MUSC, MAT["tendon"])
            bone = f"tendon_SO_{side}"
            drive_target = belly
        else:
            head, tail = Cc + O, Cc + W
            mesh_obj = finish(f"Belly_{mname}_{side}", tube_bm(line_pts(head, tail, 14), belly_radii(14, m["r"])), C_MUSC, None)
            mesh_obj.data.materials.append(muscle_material(f"Muscle_{mname}_{side}"))
            bone = f"belly_{mname}_{side}"
            drive_target = mesh_obj
        bone_specs.append((bone, head, tail))
        pending.append((bone, wemp, (tail - head).length, "VOLUME_XZX"))
        skinned.append((mesh_obj, bone))
        drive_mat = drive_target.data.materials[0]
        drive_mat.node_tree.driver_add('nodes["Activation"].inputs[0].default_value')
        fc = drive_mat.node_tree.animation_data.drivers[-1]
        d = fc.driver
        d.type, d.expression = "SCRIPTED", act_expr(mname, side)
        for var, prop in {"gx": "gaze_x_deg", "gy": "gaze_y_deg", "v": "vergence_deg"}.items():
            v = d.variables.new()
            v.name, v.type = var, "SINGLE_PROP"
            v.targets[0].id, v.targets[0].data_path = ctrl, f'["{prop}"]'

    # --- optic nerve (stretches from the apex to the back of the globe)
    head, tail = Cc + Vector((0, 5.4, 0)), Cc + nerve_target.location
    nerve = finish(f"OpticNerve_{side}", tube_bm(line_pts(head, tail, 14), [0.17 + 0.02 * math.sin(math.pi * i / 13) for i in range(14)]), C_MUSC, MAT["nerve"])
    bone_specs.append((f"nerve_{side}", head, tail))
    pending.append((f"nerve_{side}", nerve_target, (tail - head).length, "NO_VOLUME"))
    skinned.append((nerve, f"nerve_{side}"))

    # --- eyelids: shells centred on the eye; the object origin is the eye centre, so rotating about X is a blink
    # (upper edge 26 deg above the horizon, lower edge 30 deg below, when open)
    for kind, sgn, edge in (("Upper", 1, 26), ("Lower", -1, 30)):
        lim = math.sin(edge * D2R)
        bm = sphere_bm(1.28, 56, 36, keep=lambda co, sgn=sgn, lim=lim: co.y <= 0.35 and co.z * sgn / 1.28 >= lim)
        for vt in bm.verts:
            vt.co.y *= 1.2
        lo = finish(f"Lid{kind}_{side}", bm, C_LIDS, MAT["skin"])
        lo.location = Cc
        sol = lo.modifiers.new("Thickness", "SOLIDIFY")
        sol.thickness, sol.offset = 0.07, 1.0
        if kind == "Upper":
            drive(lo, "rotation_euler", f"(32*b-0.4*gy*(1-b))*{K}", {**BL, **GY}, 0)
        else:
            drive(lo, "rotation_euler", f"(-28*b-0.15*gy*(1-b))*{K}", {**BL, **GY}, 0)
        for prop in ("hide_viewport", "hide_render"):
            drive(lo, prop, "1-s", SL)

# build bones
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for name, head, tail in bone_specs:
    eb = arm.edit_bones.new(name)
    eb.head, eb.tail = head, tail
bpy.ops.object.mode_set(mode="OBJECT")
for name, target, rest, vol in pending:
    c = rig.pose.bones[name].constraints.new("STRETCH_TO")
    c.target, c.rest_length = target, rest
    c.volume = vol
    if vol != "NO_VOLUME":
        c.bulge = 1.0
for obj, bone in skinned:
    obj.vertex_groups.new(name=bone).add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    obj.parent = rig

# ----------------------------------------------------------------------------- studio: cameras, lights, world
def camera(name, loc, lens=50, rot=None, look_at=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    co = bpy.data.objects.new(name, cd)
    co.location = loc
    if rot:
        co.rotation_euler = rot
    C_STUDIO.objects.link(co)
    if look_at:
        t = empty(f"{name}_target", look_at, C_STUDIO, size=0.2, kind="PLAIN_AXES")
        tc = co.constraints.new("TRACK_TO")
        tc.target, tc.track_axis, tc.up_axis = t, "TRACK_NEGATIVE_Z", "UP_Y"
    return co


cam_front = camera("Cam_Front", (0, -20, 0), 50, (math.pi / 2, 0, 0))
camera("Cam_Top", (0, 2.2, 22), 50, (0, 0, 0))
camera("Cam_Oblique", (13, -15, 12), 50, look_at=(0, 1.6, 0))
scene.camera = cam_front


def light(name, loc, energy, size, target=(0, 0, 0)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy, ld.size = energy, size
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    C_STUDIO.objects.link(lo)
    t = empty(f"{name}_target", target, C_STUDIO, size=0.2, kind="PLAIN_AXES")
    tc = lo.constraints.new("TRACK_TO")
    tc.target, tc.track_axis, tc.up_axis = t, "TRACK_NEGATIVE_Z", "UP_Y"


light("Key", (-8, -14, 10), 1800, 8)
light("Fill", (10, -12, 2), 700, 8)
light("Rim", (0, 12, 9), 600, 6)
world = bpy.data.worlds.new("Studio")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.20, 0.21, 0.24, 1)
scene.world = world
try:
    scene.render.engine = "BLENDER_EEVEE"
except TypeError:
    scene.render.engine = "BLENDER_EEVEE_NEXT"

# ----------------------------------------------------------------------------- embed the API, demo animation
api_path = os.path.join(HERE, "artificial_eye_api.py")
with open(api_path) as f:
    api_src = f.read()
txt = bpy.data.texts.new("artificial_eye_api.py")
txt.from_string(api_src)
txt.use_module = True       # registers the sidebar panel on load when script auto-run is allowed

spec = importlib.util.spec_from_file_location("artificial_eye_api", api_path)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
api_eye = api.ArtificialEye(ctrl)

FPS = 30
F = lambda t: 1 + int(round(t * FPS))
scene.frame_start, scene.frame_end = 1, F(9.0)

cur = (0.0, 0.0)
api_eye.set_gaze(0, 0)
api_eye.key(1, ["gaze_x_deg", "gaze_y_deg"])
for t, d, gx, gy in ((0.8, 0.10, 15, 5), (1.9, 0.12, -20, -8), (3.3, 0.10, 0, 0), (5.2, 0.12, 0, 18), (6.2, 0.12, 0, -16), (7.2, 0.10, 0, 0)):
    api_eye.set_gaze(*cur)
    api_eye.key(F(t), ["gaze_x_deg", "gaze_y_deg"])
    api_eye.set_gaze(gx, gy)
    api_eye.key(F(t + d), ["gaze_x_deg", "gaze_y_deg"])
    cur = (gx, gy)
api_eye.key(F(9.0), ["gaze_x_deg", "gaze_y_deg"])

v = 0.0
api_eye._set("vergence_deg", 0)
api_eye.key(1, ["vergence_deg"])
for t, d, tgt in ((3.8, 0.8, 16), (4.9, 0.8, 0)):
    api_eye.key(F(t), ["vergence_deg"])
    api_eye._set("vergence_deg", tgt)
    api_eye.key(F(t + d), ["vergence_deg"])
api_eye.key(F(9.0), ["vergence_deg"])

api_eye.set_blink(0)
api_eye.key(1, ["blink"])
for t in (1.4, 4.4, 6.6):
    for dt, val in ((0, 0), (3, 1), (5, 1), (9, 0)):
        api_eye.set_blink(val)
        api_eye.key(F(t) + dt, ["blink"])
api_eye.set_blink(0)
api_eye.key(F(9.0), ["blink"])

api_eye.set_pupil(4.0)
api_eye.key(1, ["pupil_mm"])
for t, d, mm in ((4.0, 1.5, 6.0), (6.8, 1.0, 3.2)):
    api_eye.key(F(t), ["pupil_mm"])
    api_eye.set_pupil(mm)
    api_eye.key(F(t + d), ["pupil_mm"])
api_eye.key(F(9.0), ["pupil_mm"])
api_eye.set_gaze(0, 0, 0)
api_eye.set_blink(0)
api_eye.set_pupil(4.0)
scene.frame_set(1)

bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True)
print("Saved", OUT)
