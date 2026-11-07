"""
LAST SEEN - master Blender script (Blender 5.2 LTS, Cycles).

Downloads free realistic assets, builds the whole 2-minute film as one .blend,
and renders test frames or the final image sequence.

    Build everything and render one test frame:
        blender -b -P last_seen_master.py

    Render chosen test frames from the built file:
        blender -b last_seen.blend -P last_seen_master.py -- --frames 600,1500 --scale 50 --samples 64

    Final render on the NVIDIA PC (skips frames that already exist, so it can be stopped and resumed):
        blender -b last_seen.blend -P last_seen_master.py -- --final

    Join the finished frames into an MP4:
        blender -b last_seen.blend -P last_seen_master.py -- --encode

Asset sources (all free, see CREDITS.txt written next to this script):
    Blender Foundation BMW demo scene, Poly Haven (CC0), Microsoft Rocketbox (MIT).
"""
import math
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import blf
import bmesh
import bpy
import imbuf
import numpy as np
from bpy_extras import anim_utils
from mathutils import Euler, Matrix, Vector
from mathutils import noise as mnoise
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
def _script_dir():
    try:
        p = Path(__file__).resolve()
        if p.exists():
            return p.parent
    except NameError:
        pass
    for text in bpy.data.texts:
        if text.filepath.endswith("last_seen_master.py"):
            return Path(bpy.path.abspath(text.filepath)).resolve().parent
    raise RuntimeError("Save last_seen_master.py to disk before running it.")


PROJECT = _script_dir()
ASSETS = PROJECT / "assets"
GENERATED = ASSETS / "generated"
BLEND_PATH = PROJECT / "last_seen.blend"
FRAMES_DIR = PROJECT / "frames"
TEST_DIR = PROJECT / "test_frames"

USER_AGENT = "LastSeenBlenderSetup/1.0"
FPS = 24
DURATION_S = 120
LAST_FRAME = DURATION_S * FPS
RESOLUTION = (1920, 1080)
DEFAULT_TEST_FRAME = 33 * FPS + 1   # the girl in the road, the first jump scare
FINAL_SAMPLES = 64    # plenty with the denoiser; this is the main speed-versus-quality dial

POLYHAVEN_HDRIS = {"night": "kloppenheim_07_puresky", "morning": "kloofendal_misty_morning_puresky"}
POLYHAVEN_TEXTURES = ["asphalt_02", "stony_dirt_path", "forest_leaves_04", "mud_forest"]
POLYHAVEN_MODELS = [
    "island_tree_01", "island_tree_02",
    "dead_tree_trunk", "dead_tree_trunk_02", "tree_stump_01", "fir_sapling",
    "fern_02", "grass_medium_01", "shrub_02",
    "root_cluster_01", "pine_roots", "rock_moss_set_01", "dry_branches_medium_01",
    "rusted_spade_01", "concrete_road_barrier",
]

BMW_URL = "https://download.blender.org/demo/test/BMW27_2.blend.zip"
ROCKETBOX_RAW = "https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master"
ROCKETBOX_AVATARS = {
    "Male_Adult_07": ("Adults", "m013", ["body", "head"], True),
    "Female_Child_02": ("Children", "cf011", ["body", "head"], False),
    "Police_Male_01": ("Professions", "m220", ["body", "head", "equipment", "pistol"], False),
}
ROCKETBOX_ANIMS = {
    "static": ["m_idle_nervous_01", "m_sit_chair_idle_neutral_01", "m_crouch_idle", "m_idle_look_around_01",
               "m_cell_phone_textmessage", "f_idle_neutral_01", "f_idle_nervous_01"],
    "xy": ["m_walk_slow_01", "m_walk_neutral_01", "m_walk_bruised"],
}


def log(*args):
    print("[LAST SEEN]", *args, flush=True)


# ---------------------------------------------------------------------------
# Asset download
# ---------------------------------------------------------------------------
def download(url, dest, min_bytes=1024):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            if tmp.stat().st_size < min_bytes:
                raise IOError(f"file too small ({tmp.stat().st_size} bytes)")
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, IOError, TimeoutError) as error:
            last_error = error
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"Download failed: {url}\n  {last_error}")


def polyhaven_files(asset_id):
    cache = ASSETS / "polyhaven" / "_index" / f"{asset_id}.json"
    download(f"https://api.polyhaven.com/files/{asset_id}", cache, min_bytes=16)
    return json.loads(cache.read_text(encoding="utf-8"))


def fetch_all_assets():
    jobs = []
    # Skies
    for asset_id in POLYHAVEN_HDRIS.values():
        info = polyhaven_files(asset_id)
        jobs.append((info["hdri"]["2k"]["hdr"]["url"], ASSETS / "polyhaven" / "hdri" / f"{asset_id}.hdr"))
    # PBR textures
    for asset_id in POLYHAVEN_TEXTURES:
        info = polyhaven_files(asset_id)
        for map_key, short in (("Diffuse", "diff"), ("nor_gl", "nor"), ("Rough", "rough")):
            if map_key in info and "2k" in info[map_key]:
                entry = info[map_key]["2k"].get("jpg") or info[map_key]["2k"].get("png")
                suffix = Path(entry["url"].split("?")[0]).suffix
                jobs.append((entry["url"], ASSETS / "polyhaven" / "tex" / asset_id / f"{short}{suffix}"))
    # 3D models
    for asset_id in POLYHAVEN_MODELS:
        info = polyhaven_files(asset_id)
        entry = info["gltf"]["1k"]["gltf"]
        root = ASSETS / "polyhaven" / "models" / asset_id
        jobs.append((entry["url"], root / Path(entry["url"].split("?")[0]).name))
        for rel, sub in entry.get("include", {}).items():
            jobs.append((sub["url"], root / rel))
    # Rocketbox people
    for name, (group, prefix, parts, has_opacity) in ROCKETBOX_AVATARS.items():
        folder = ASSETS / "rocketbox" / name
        base = f"{ROCKETBOX_RAW}/Assets/Avatars/{group}/{name}"
        jobs.append((f"{base}/Export/{name}.fbx", folder / f"{name}.fbx"))
        for part in parts:
            for kind in ("color", "normal", "specular"):
                jobs.append((f"{base}/Textures/{prefix}_{part}_{kind}.tga", folder / f"{prefix}_{part}_{kind}.tga"))
        if has_opacity:
            jobs.append((f"{base}/Textures/{prefix}_opacity_color.tga", folder / f"{prefix}_opacity_color.tga"))
    for mode, clips in ROCKETBOX_ANIMS.items():
        for clip in clips:
            url = f"{ROCKETBOX_RAW}/Assets/Animations/all_animations_max_motextr_{mode}/{clip}.max.fbx"
            jobs.append((url, ASSETS / "rocketbox" / "_anim" / f"{clip}.fbx"))
    jobs.append((f"{ROCKETBOX_RAW}/LICENSE.md", ASSETS / "rocketbox" / "LICENSE.txt"))
    # The BMW
    bmw_zip = ASSETS / "car_bmw27" / "BMW27_2.blend.zip"
    jobs.append((BMW_URL, bmw_zip))

    todo = [job for job in jobs if not (Path(job[1]).exists() and Path(job[1]).stat().st_size >= 1024)]
    log(f"{len(jobs)} asset files, {len(todo)} to download")
    done = [0]

    def worker(job):
        result = download(job[0], job[1], min_bytes=64)
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(todo):
            log(f"  downloaded {done[0]}/{len(todo)}")
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, todo))

    bmw_blend = ASSETS / "car_bmw27" / "bmw27" / "bmw27_gpu.blend"
    if not bmw_blend.exists():
        with zipfile.ZipFile(bmw_zip) as archive:
            archive.extractall(ASSETS / "car_bmw27")
    log("assets ready")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fr(seconds):
    """Frame number for a story time in seconds."""
    return int(round(seconds * FPS)) + 1


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def get_collection(name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(coll)
    return coll


def add_object(name, data, coll):
    obj = bpy.data.objects.new(name, data)
    coll.objects.link(obj)
    return obj


def move_to(obj, coll):
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    coll.objects.link(obj)


def mesh_object(name, verts, faces, coll, smooth=True):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if smooth:
        mesh.shade_smooth()
    return add_object(name, mesh, coll)


def look_at(location, target, roll_deg=0.0):
    """Rotation (quaternion) that aims a camera or spot at target."""
    direction = Vector(target) - Vector(location)
    quat = direction.to_track_quat("-Z", "Y")
    if roll_deg:
        quat = quat @ Euler((0, 0, math.radians(roll_deg))).to_quaternion()
    return quat


def curves_of(owner, prop, index=-1):
    """The animation curves behind one property of any object, light, node socket or setting."""
    data = owner.id_data.animation_data
    bag = anim_utils.action_get_channelbag_for_slot(data.action, data.action_slot)
    path = owner.path_from_id(prop)
    return [c for c in bag.fcurves if c.data_path == path and (index < 0 or c.array_index == index)]


def key_values(owner, prop, keys, kind="LINEAR", index=-1):
    """keys = [(frame, value), ...] on any animatable property. kind is CONSTANT, LINEAR or BEZIER."""
    for frame, value in keys:
        if index >= 0:
            getattr(owner, prop)[index] = value
        else:
            setattr(owner, prop, value)
        owner.keyframe_insert(prop, frame=frame, index=index)
    frames = {float(key[0]) for key in keys}
    for curve in curves_of(owner, prop, index):
        for point in curve.keyframe_points:
            if point.co.x in frames:
                point.interpolation = kind


def key_visible(obj, windows):
    """Show obj only inside the given (first, last) frame windows."""
    keys = [(1, True)]
    for first, last in windows:
        keys += [(first, False), (last + 1, True)]
    keys = sorted({frame: hidden for frame, hidden in keys}.items())
    key_values(obj, "hide_render", keys, kind="CONSTANT")


class Nodes:
    """Keeps node-tree building code short and readable."""

    def __init__(self, tree, clear=True):
        self.tree = tree
        if clear:
            tree.nodes.clear()

    def new(self, kind, **settings):
        node = self.tree.nodes.new(kind)
        for key, value in settings.items():
            if key in node.inputs:
                socket = node.inputs[key]
                if isinstance(value, bpy.types.NodeSocket):
                    self.tree.links.new(value, socket)
                else:
                    socket.default_value = value
            else:
                setattr(node, key, value)
        return node

    def wire(self, source, target):
        self.tree.links.new(source, target)

    def _feed(self, socket, value):
        if isinstance(value, bpy.types.NodeSocket):
            self.tree.links.new(value, socket)
        else:
            socket.default_value = value

    def math(self, operation, a, b=None, c=None, clamp=False):
        node = self.new("ShaderNodeMath", operation=operation, use_clamp=clamp)
        for index, value in enumerate((a, b, c)):
            if value is not None:
                self._feed(node.inputs[index], value)
        return node.outputs[0]

    def mix(self, factor, a, b, blend="MIX"):
        node = self.new("ShaderNodeMixRGB", blend_type=blend)
        for index, value in enumerate((factor, a, b)):
            self._feed(node.inputs[index], value)
        return node.outputs[0]

    def map_range(self, value, from_min, from_max, to_min=0.0, to_max=1.0, smooth=False):
        node = self.new("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP" if smooth else "LINEAR", clamp=True)
        self._feed(node.inputs["Value"], value)
        node.inputs["From Min"].default_value = from_min
        node.inputs["From Max"].default_value = from_max
        node.inputs["To Min"].default_value = to_min
        node.inputs["To Max"].default_value = to_max
        return node.outputs[0]

    def shader_mix(self, factor, a, b):
        node = self.new("ShaderNodeMixShader")
        self._feed(node.inputs[0], factor)
        self.wire(a, node.inputs[1])
        self.wire(b, node.inputs[2])
        return node.outputs[0]

    def time(self):
        """Value node that outputs story time in seconds."""
        node = self.new("ShaderNodeValue")
        curve = node.outputs[0].driver_add("default_value")
        curve.driver.expression = f"frame/{FPS}"
        return node.outputs[0]


def load_image(path, non_color=False):
    image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    return image


def tex_file(asset, short):
    for candidate in (ASSETS / "polyhaven" / "tex" / asset).glob(short + ".*"):
        if candidate.suffix != ".part":
            return candidate
    raise FileNotFoundError(f"Missing texture {asset}/{short}. Run the script online once to download assets.")


def pbr_nodes(nb, asset, vector):
    """Returns (color, roughness, normal colour) sockets for a Poly Haven texture."""
    color = nb.new("ShaderNodeTexImage", image=load_image(tex_file(asset, "diff")))
    rough = nb.new("ShaderNodeTexImage", image=load_image(tex_file(asset, "rough"), non_color=True))
    normal = nb.new("ShaderNodeTexImage", image=load_image(tex_file(asset, "nor"), non_color=True))
    for node in (color, rough, normal):
        nb.wire(vector, node.inputs["Vector"])
    return color.outputs["Color"], rough.outputs["Color"], normal.outputs["Color"]


def rain_ripples(nb, vector, seconds, scale, rate, phase=0.0):
    """Expanding ring height pattern: the rings raindrops make on standing water."""
    cells = nb.new("ShaderNodeTexVoronoi", feature="F1", voronoi_dimensions="2D", Scale=scale)
    nb.wire(vector, cells.inputs["Vector"])
    distance, cell_random = cells.outputs["Distance"], cells.outputs["Color"]
    age = nb.math("FRACT", nb.math("ADD", nb.math("MULTIPLY", seconds, rate), nb.math("MULTIPLY", cell_random, 13.7 + phase)))
    offset = nb.math("SUBTRACT", distance, nb.math("MULTIPLY", age, 0.75))
    envelope = nb.math("MAXIMUM", nb.math("SUBTRACT", 1.0, nb.math("DIVIDE", nb.math("ABSOLUTE", offset), 0.13)), 0.0)
    fade = nb.math("POWER", nb.math("SUBTRACT", 1.0, age), 2.0)
    wave = nb.math("SINE", nb.math("MULTIPLY", offset, 62.0))
    return nb.math("MULTIPLY", nb.math("MULTIPLY", wave, envelope), fade)


def drop_height(nb, vector, scale, radius):
    """Round water beads: returns (height, mask) sockets."""
    cells = nb.new("ShaderNodeTexVoronoi", feature="F1", voronoi_dimensions="3D", Scale=scale, Randomness=1.0)
    nb.wire(vector, cells.inputs["Vector"])
    size = nb.math("MULTIPLY", nb.math("POWER", cells.outputs["Color"], 2.2), radius)
    inside = nb.math("SUBTRACT", nb.math("MULTIPLY", size, size), nb.math("MULTIPLY", cells.outputs["Distance"], cells.outputs["Distance"]))
    height = nb.math("SQRT", nb.math("MAXIMUM", inside, 0.0))
    mask = nb.math("GREATER_THAN", inside, 0.0)
    return height, mask


# ---------------------------------------------------------------------------
# Story layout: where the roads go and how the car moves along them
# ---------------------------------------------------------------------------
class RoadPath:
    def __init__(self, control_points):
        points = [np.array(p, dtype=float) for p in control_points]
        samples = []
        for i in range(len(points) - 1):
            p0, p1 = points[max(i - 1, 0)], points[i]
            p2, p3 = points[i + 1], points[min(i + 2, len(points) - 1)]
            count = max(int(np.linalg.norm(p2 - p1) * 2), 4)
            for j in range(count):
                t = j / count
                samples.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                                      + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
        samples.append(points[-1])
        self.xy = np.array(samples)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))])
        self.length = float(self.s[-1])

    def pos(self, s):
        return Vector((float(np.interp(s, self.s, self.xy[:, 0])), float(np.interp(s, self.s, self.xy[:, 1])), 0.0))

    def tangent(self, s):
        return (self.pos(s + 0.6) - self.pos(s - 0.6)).normalized()

    def right(self, s):
        t = self.tangent(s)
        return Vector((t.y, -t.x, 0.0))

    def closest_s(self, x, y):
        return float(self.s[np.argmin(np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y))])

    def distance_field(self, x, y, chunk=8000):
        pts = self.xy[::2]
        result = np.empty(x.shape, dtype=float)
        fx, fy = x.ravel(), y.ravel()
        out = result.ravel()
        for start in range(0, fx.size, chunk):
            dx = fx[start:start + chunk, None] - pts[:, 0]
            dy = fy[start:start + chunk, None] - pts[:, 1]
            out[start:start + chunk] = np.sqrt((dx * dx + dy * dy).min(axis=1))
        return out.reshape(x.shape)


HIGHWAY_CTRL = [(10, -560), (6, -420), (0, -280), (0, -100), (0, 60), (0, 130), (0, 330)]
FOREST_CTRL = [(0, 86), (3, 97), (14, 108), (45, 118), (90, 128), (130, 150), (160, 185), (175, 230), (178, 280), (170, 330), (150, 380)]
CAR_CTRL = [(-1.9, -300), (-1.9, -100), (-1.9, 40), (-1.9, 70), (-0.8, 82), (3, 97), (14, 108), (45, 118), (90, 128),
            (130, 150), (160, 185), (175, 230), (178, 280), (170, 330), (150, 380)]
HIGHWAY_HALF_WIDTH = 4.2
FOREST_HALF_WIDTH = 2.8

# (story seconds, metres per second)
SPEED_KEYS = [(0, 19.4), (12, 19.4), (15.5, 0), (19, 0), (22, 10), (33, 11), (34.3, 0), (38, 0), (39.5, 4), (55, 4), (59, 0), (120, 0)]
JUNCTION_STOP_Y = 74.0
FLASHBACK = (80.0, 88.0)          # story seconds of the "one year ago" sequence
FLASHBACK_IMPACT = 81.9


class Layout:
    def __init__(self):
        self.highway = RoadPath(HIGHWAY_CTRL)
        self.forest = RoadPath(FOREST_CTRL)
        self.car_path = RoadPath(CAR_CTRL)
        times = (np.arange(LAST_FRAME + 2) - 1) / FPS
        speeds = np.interp(times, [k[0] for k in SPEED_KEYS], [k[1] for k in SPEED_KEYS])
        travelled = np.concatenate([[0.0], np.cumsum(speeds[1:] / FPS)])
        stop_index = int(round(15.5 * FPS)) + 1
        self.car_s = travelled + (self.car_path.closest_s(-1.9, JUNCTION_STOP_Y) - travelled[stop_index])
        self.parked_s = float(self.car_s[-1])
        parked = self.car_path.pos(self.parked_s)
        self.parked_pos = parked
        forest_s = self.forest.closest_s(parked.x, parked.y)
        self.forest_parked_s = forest_s
        self.big_tree = self.forest.pos(forest_s + 5.0) + self.forest.right(forest_s + 5.0) * 15.0
        self.impact_s = self.parked_s + 3.0

    def edge_distance(self, x, y):
        return np.minimum(self.highway.distance_field(x, y) - HIGHWAY_HALF_WIDTH,
                          self.forest.distance_field(x, y) - FOREST_HALF_WIDTH)

    def ground_z(self, x, y):
        edge = self.edge_distance(x, y)
        rolling = (1.4 * np.sin(x * 0.029 + 1.3 * np.sin(y * 0.016)) + 1.0 * np.sin(y * 0.041 + x * 0.013 + 2.0)
                   + 0.45 * np.sin(x * 0.11 + y * 0.07) + 0.22 * np.sin(x * 0.31 - y * 0.23 + 1.0)
                   + 0.12 * np.sin(x * 0.71 + y * 0.57))
        ditch = -0.16 * np.exp(-((edge - 3.6) / 1.5) ** 2)
        # keep the big tree's clearing fairly level so people stand on it believably
        clearing = smoothstep(4.0, 16.0, np.hypot(x - self.big_tree.x, y - self.big_tree.y))
        return (rolling * smoothstep(2.0, 18.0, edge) * clearing + ditch - 0.012), edge

    def height(self, x, y):
        z, _ = self.ground_z(np.array([float(x)]), np.array([float(y)]))
        return float(z[0])

    def distance_to_walk(self, x, y):
        """Distance from the straight line he walks between the parked car and the big tree."""
        ax, ay = self.parked_pos.x, self.parked_pos.y
        bx, by = self.big_tree.x, self.big_tree.y
        dx, dy = bx - ax, by - ay
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        return np.hypot(x - (ax + t * dx), y - (ay + t * dy))

    def car_frame(self, frame):
        """Car position, heading and speed at a frame, including the flashback."""
        seconds = (frame - 1) / FPS
        index = int(np.clip(frame, 0, len(self.car_s) - 1))
        if FLASHBACK[0] <= seconds < FLASHBACK[1]:
            travelled = min(seconds, FLASHBACK_IMPACT) - FLASHBACK[0]
            s = self.impact_s - (FLASHBACK_IMPACT - FLASHBACK[0]) * 14.0 + travelled * 14.0 - 2.6
            speed = 14.0 if seconds < FLASHBACK_IMPACT else 0.0
        else:
            s = float(self.car_s[index])
            previous = float(self.car_s[max(index - 1, 0)])
            speed = (s - previous) * FPS
        return self.car_path.pos(s), self.car_path.tangent(s), speed


# ---------------------------------------------------------------------------
# Road and forest floor
# ---------------------------------------------------------------------------
def build_road_material(name, lane_lines):
    material = bpy.data.materials.new(name)
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled")
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    uv = nb.new("ShaderNodeUVMap", uv_map="RoadUV").outputs[0]
    position = nb.new("ShaderNodeNewGeometry").outputs["Position"]
    seconds = nb.time()
    edge = nb.new("ShaderNodeAttribute", attribute_name="edge").outputs["Factor"]

    tile_road = nb.new("ShaderNodeMapping", Scale=(1 / 3.2, 1 / 3.2, 1.0))
    nb.wire(uv, tile_road.inputs["Vector"])
    tile_dirt = nb.new("ShaderNodeMapping", Scale=(1 / 2.6, 1 / 2.6, 1.0))
    nb.wire(uv, tile_dirt.inputs["Vector"])
    road_color, road_rough, road_normal = pbr_nodes(nb, "asphalt_02", tile_road.outputs[0])
    dirt_color, dirt_rough, dirt_normal = pbr_nodes(nb, "stony_dirt_path", tile_dirt.outputs[0])

    wear = nb.new("ShaderNodeTexNoise", Scale=1.3, Detail=7.0, Roughness=0.6)
    nb.wire(position, wear.inputs["Vector"])
    dirt_mask = nb.map_range(nb.math("ADD", edge, nb.math("MULTIPLY", wear.outputs[0], 0.34)), 0.97, 1.12, smooth=True)

    # Large shallow puddles collect where the noise is high, more toward the road edges.
    puddle_noise = nb.new("ShaderNodeTexNoise", Scale=0.42, Detail=5.0, Roughness=0.55)
    nb.wire(position, puddle_noise.inputs["Vector"])
    puddle_value = nb.math("ADD", puddle_noise.outputs[0], nb.math("MULTIPLY", edge, 0.08))
    puddle = nb.map_range(puddle_value, 0.58, 0.64, smooth=True)

    base = nb.mix(dirt_mask, road_color, dirt_color)
    wet_tint = nb.math("ADD", nb.math("MULTIPLY", puddle, -0.18), 0.52)
    base = nb.mix(1.0, base, (1, 1, 1, 1), blend="MULTIPLY")
    gain = nb.new("ShaderNodeVectorMath", operation="SCALE")
    nb.wire(base, gain.inputs[0])
    nb.wire(wet_tint, gain.inputs["Scale"])
    base = gain.outputs[0]
    rough = nb.mix(dirt_mask, road_rough, dirt_rough)
    rough = nb.math("ADD", nb.math("MULTIPLY", rough, 0.42), 0.06)
    rough = nb.mix(puddle, rough, (0.025, 0.025, 0.025, 1))

    if lane_lines:
        across = nb.new("ShaderNodeSeparateXYZ")
        nb.wire(uv, across.inputs[0])
        u, v = across.outputs[0], across.outputs[1]
        centre = nb.math("LESS_THAN", nb.math("ABSOLUTE", u), 0.075)
        dash = nb.math("LESS_THAN", nb.math("FRACT", nb.math("DIVIDE", v, 9.0)), 0.4)
        centre = nb.math("MULTIPLY", centre, dash)
        side = nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SUBTRACT", nb.math("ABSOLUTE", u), HIGHWAY_HALF_WIDTH - 0.3)), 0.06)
        paint = nb.math("MINIMUM", nb.math("MAXIMUM", centre, side), nb.map_range(wear.outputs[0], 0.34, 0.47))
        base = nb.mix(paint, base, (0.52, 0.52, 0.5, 1))

    nb.wire(base, bsdf.inputs["Base Color"])
    nb.wire(rough, bsdf.inputs["Roughness"])

    flat = nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", puddle, 0.85))
    normal_map = nb.new("ShaderNodeNormalMap")
    nb.wire(nb.mix(dirt_mask, road_normal, dirt_normal), normal_map.inputs["Color"])
    nb.wire(flat, normal_map.inputs["Strength"])
    ripple = nb.new("ShaderNodeSeparateXYZ")
    nb.wire(position, ripple.inputs[0])
    ground_xy = nb.new("ShaderNodeCombineXYZ")
    nb.wire(ripple.outputs[0], ground_xy.inputs[0])
    nb.wire(ripple.outputs[1], ground_xy.inputs[1])
    rings = nb.math("ADD", rain_ripples(nb, ground_xy.outputs[0], seconds, 5.5, 1.5),
                    rain_ripples(nb, ground_xy.outputs[0], seconds, 9.5, 1.9, phase=4.0))
    bump = nb.new("ShaderNodeBump", Strength=0.75, Distance=0.006)
    nb.wire(rings, bump.inputs["Height"])
    nb.wire(normal_map.outputs[0], bump.inputs["Normal"])
    nb.wire(bump.outputs[0], bsdf.inputs["Normal"])
    return material


def build_ground_material():
    material = bpy.data.materials.new("ForestFloor")
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled")
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    position = nb.new("ShaderNodeNewGeometry").outputs["Position"]
    tile_a = nb.new("ShaderNodeMapping", Scale=(1 / 4.1, 1 / 4.1, 1 / 4.1))
    nb.wire(position, tile_a.inputs["Vector"])
    tile_b = nb.new("ShaderNodeMapping", Scale=(1 / 3.3, 1 / 3.3, 1 / 3.3), Rotation=(0, 0, 0.7))
    nb.wire(position, tile_b.inputs["Vector"])
    leaf_color, leaf_rough, leaf_normal = pbr_nodes(nb, "forest_leaves_04", tile_a.outputs[0])
    mud_color, mud_rough, mud_normal = pbr_nodes(nb, "mud_forest", tile_b.outputs[0])
    patches = nb.new("ShaderNodeTexNoise", Scale=0.16, Detail=6.0, Roughness=0.6)
    nb.wire(position, patches.inputs["Vector"])
    near_road = nb.new("ShaderNodeAttribute", attribute_name="near_road").outputs["Factor"]
    mud = nb.math("MAXIMUM", nb.map_range(patches.outputs[0], 0.47, 0.60, smooth=True), near_road, clamp=True)
    color = nb.mix(mud, leaf_color, mud_color)
    wet = nb.new("ShaderNodeVectorMath", operation="SCALE", Scale=0.5)
    nb.wire(color, wet.inputs[0])
    nb.wire(wet.outputs[0], bsdf.inputs["Base Color"])
    rough = nb.math("ADD", nb.math("MULTIPLY", nb.mix(mud, leaf_rough, mud_rough), 0.55), 0.08)
    nb.wire(rough, bsdf.inputs["Roughness"])
    normal_map = nb.new("ShaderNodeNormalMap", Strength=1.4)
    nb.wire(nb.mix(mud, leaf_normal, mud_normal), normal_map.inputs["Color"])
    nb.wire(normal_map.outputs[0], bsdf.inputs["Normal"])
    return material


def build_road(name, path, half_width, material, coll, z_offset, rough_edges):
    across = 12
    step = 1.0
    stations = np.arange(0.0, path.length, step)
    verts, faces, edge_values, uvs = [], [], [], []
    for row, s in enumerate(stations):
        centre, right = path.pos(s), path.right(s)
        wobble = 0.0
        if rough_edges:
            wobble = 0.22 * math.sin(s * 0.37) + 0.14 * math.sin(s * 1.13 + 1.0) + 0.08 * math.sin(s * 2.9)
        for k in range(across + 1):
            u = -1.0 + 2.0 * k / across
            width = half_width + wobble * abs(u)
            p = centre + right * (u * width)
            verts.append((p.x, p.y, 0.035 * (1.0 - u * u) + z_offset))
            edge_values.append(abs(u))
            uvs.append((u * width, float(s)))
    columns = across + 1
    for row in range(len(stations) - 1):
        for k in range(across):
            a = row * columns + k
            faces.append((a, a + 1, a + columns + 1, a + columns))
    obj = mesh_object(name, verts, faces, coll)
    mesh = obj.data
    mesh.materials.append(material)
    attribute = mesh.attributes.new("edge", "FLOAT", "POINT")
    attribute.data.foreach_set("value", edge_values)
    uv_layer = mesh.uv_layers.new(name="RoadUV")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            uv_layer.data[loop_index].uv = uvs[mesh.loops[loop_index].vertex_index]
    return obj


def build_terrain(layout, coll):
    spacing = 2.5
    xs = np.arange(-240.0, 440.0, spacing)
    ys = np.arange(-620.0, 520.0, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    height, edge = layout.ground_z(grid_x, grid_y)
    keep = edge < 150.0
    verts = np.column_stack([grid_x.ravel(), grid_y.ravel(), height.ravel()])
    columns = xs.size
    index = np.arange(grid_x.size).reshape(grid_x.shape)
    a, b, c, d = index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]
    face_keep = keep[:-1, :-1] | keep[:-1, 1:] | keep[1:, 1:] | keep[1:, :-1]
    faces = np.column_stack([a[face_keep], b[face_keep], c[face_keep], d[face_keep]])
    obj = mesh_object("ForestFloor", verts.tolist(), faces.tolist(), coll)
    near = 1.0 - smoothstep(0.0, 5.0, edge)
    attribute = obj.data.attributes.new("near_road", "FLOAT", "POINT")
    attribute.data.foreach_set("value", near.ravel().astype(np.float32))
    obj.data.materials.append(build_ground_material())
    log(f"terrain: {len(obj.data.vertices)} points, {columns} columns")
    return obj


# ---------------------------------------------------------------------------
# Imported Poly Haven models and the forest
# ---------------------------------------------------------------------------
def import_polyhaven_model(asset_id, coll):
    folder = ASSETS / "polyhaven" / "models" / asset_id
    gltf = next(folder.glob("*.gltf"))
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(gltf))
    imported = [o for o in bpy.data.objects if o not in before]
    for obj in imported:
        move_to(obj, coll)
    return imported


def wetten_material(material, amount=0.55):
    """Rain-soaked look for imported models: darker and shinier."""
    if material is None or not material.node_tree:
        return
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        rough = node.inputs["Roughness"]
        if rough.is_linked:
            scale = material.node_tree.nodes.new("ShaderNodeMath")
            scale.operation = "MULTIPLY"
            scale.inputs[1].default_value = amount
            material.node_tree.links.new(rough.links[0].from_socket, scale.inputs[0])
            material.node_tree.links.new(scale.outputs[0], rough)
        else:
            rough.default_value *= amount


def scatter(name, source, xyz, coll, scale=(0.8, 1.2), lean=0.03, seed=1):
    """Instance `source` at every point. Instances share memory, which keeps VRAM low."""
    if len(xyz) == 0:
        return None
    group = bpy.data.node_groups.new(f"Scatter_{name}", "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    inputs, outputs = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    info.transform_space = "ORIGINAL"
    info.inputs["Object"].default_value = source
    info.inputs["As Instance"].default_value = True
    on_points = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(inputs.outputs[0], on_points.inputs["Points"])
    links.new(info.outputs["Geometry"], on_points.inputs["Instance"])
    size = nodes.new("FunctionNodeRandomValue")
    size.data_type = "FLOAT"
    size.inputs["Min"].default_value = scale[0]
    size.inputs["Max"].default_value = scale[1]
    size.inputs["Seed"].default_value = seed
    links.new(size.outputs[0], on_points.inputs["Scale"])
    turn = nodes.new("FunctionNodeRandomValue")
    turn.data_type = "FLOAT_VECTOR"
    turn.inputs["Min"].default_value = (-lean, -lean, 0.0)
    turn.inputs["Max"].default_value = (lean, lean, math.tau)
    turn.inputs["Seed"].default_value = seed + 17
    links.new(turn.outputs[0], on_points.inputs["Rotation"])
    links.new(on_points.outputs[0], outputs.inputs[0])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(p) for p in xyz], [], [])
    obj = add_object(name, mesh, coll)
    obj.modifiers.new("Scatter", "NODES").node_group = group
    obj.cycles.use_camera_cull = True
    obj.cycles.use_distance_cull = True
    return obj


def jitter_grid(layout, spacing, rng, edge_min, edge_max, bounds=(-240.0, 440.0, -620.0, 520.0), clearing=3.2):
    xs = np.arange(bounds[0], bounds[1], spacing)
    ys = np.arange(bounds[2], bounds[3], spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    x = grid_x.ravel() + rng.uniform(-0.45, 0.45, grid_x.size) * spacing
    y = grid_y.ravel() + rng.uniform(-0.45, 0.45, grid_x.size) * spacing
    z, edge = layout.ground_z(x, y)
    keep = (edge > edge_min) & (edge < edge_max) & (layout.distance_to_walk(x, y) > clearing)
    return np.column_stack([x[keep], y[keep], z[keep]])


def source_models(asset_id, coll):
    """Import a model once and return its meshes by name, soaked by the rain."""
    objects = {o.name: o for o in import_polyhaven_model(asset_id, coll) if o.type == "MESH"}
    for obj in objects.values():
        obj.location = (0.0, 0.0, 0.0)
        for material in obj.data.materials:
            wetten_material(material)
    return objects


def build_forest(layout):
    sources = get_collection("_Sources")
    forest = get_collection("Forest")
    rng = np.random.default_rng(7)
    models = {asset: source_models(asset, sources) for asset in (
        "island_tree_01", "island_tree_02", "shrub_02", "fern_02", "grass_medium_01",
        "rock_moss_set_01", "dead_tree_trunk", "dead_tree_trunk_02", "dry_branches_medium_01", "tree_stump_01",
        "fir_sapling", "root_cluster_01", "pine_roots")}

    def spread(points, names, asset, scale, lean, seed):
        """Share the points between several variants of one model."""
        options = list(names)
        picks = rng.integers(0, len(options), len(points))
        for i, name in enumerate(options):
            scatter(f"{asset}_{name}", models[asset][name], points[picks == i], forest, scale=scale, lean=lean, seed=seed + i)

    # Trees: a dense canopy near the road that thins into fog further away.
    near_trees = jitter_grid(layout, 7.0, rng, 3.8, 34.0)
    far_trees = jitter_grid(layout, 10.5, rng, 34.0, 95.0)
    tree_kinds = (("island_tree_01", "island_tree_01_LOD0"), ("island_tree_02", "island_tree_02_LOD0"))
    for group, points, scale in (("near", near_trees, (2.3, 3.5)), ("far", far_trees, (2.8, 4.2))):
        picks = rng.integers(0, len(tree_kinds), len(points))
        for i, (asset, name) in enumerate(tree_kinds):
            scatter(f"trees_{group}_{asset}", models[asset][name], points[picks == i], forest, scale=scale, lean=0.05, seed=11 + i)
    log(f"forest: {len(near_trees) + len(far_trees)} trees")

    # Undergrowth is densest where the camera can actually see it.
    spread(jitter_grid(layout, 3.4, rng, 1.6, 26.0), ("shrub_02_a", "shrub_02_b", "shrub_02_c", "shrub_02_d"),
           "shrub_02", (0.75, 1.5), 0.05, 31)
    spread(jitter_grid(layout, 2.3, rng, 1.2, 22.0), ("fern_02_a", "fern_02_b", "fern_02_c", "fern_02_d"),
           "fern_02", (1.0, 1.9), 0.08, 41)
    spread(jitter_grid(layout, 0.95, rng, 0.25, 8.0, clearing=0.9),
           ("grass_medium_01_large_a_LOD0", "grass_medium_01_large_b_LOD0", "grass_medium_01_large_c_LOD0",
            "grass_medium_01_tall_a_LOD0", "grass_medium_01_tall_b_LOD0", "grass_medium_01_mid_a_LOD0"),
           "grass_medium_01", (1.6, 3.0), 0.1, 51)
    spread(jitter_grid(layout, 7.5, rng, 2.2, 30.0), ("fir_sapling_a", "fir_sapling_b", "fir_sapling_c"),
           "fir_sapling", (0.8, 1.9), 0.06, 61)
    rocks = tuple(models["rock_moss_set_01"].keys())
    spread(jitter_grid(layout, 15.0, rng, 2.4, 45.0), rocks, "rock_moss_set_01", (0.16, 0.62), 0.4, 71)
    spread(jitter_grid(layout, 10.0, rng, 1.4, 32.0), tuple(models["dry_branches_medium_01"].keys()),
           "dry_branches_medium_01", (0.9, 2.2), 0.15, 81)
    spread(jitter_grid(layout, 26.0, rng, 3.0, 40.0), ("dead_tree_trunk_02",), "dead_tree_trunk_02", (0.9, 1.9), 0.12, 91)
    spread(jitter_grid(layout, 34.0, rng, 3.0, 40.0), ("tree_stump_01",), "tree_stump_01", (0.7, 1.3), 0.05, 95)
    return models


# ---------------------------------------------------------------------------
# Weather and light
# ---------------------------------------------------------------------------
def build_rain(coll):
    streak_verts, streak_faces = [], []
    sides = 3
    radius, length = 0.0011, 0.2
    for ring, (z, r, lean) in enumerate(((0.0, 0.25, 0.0), (length * 0.25, 1.0, 0.004), (length, 0.15, 0.018))):
        for k in range(sides):
            angle = math.tau * k / sides
            streak_verts.append((lean + r * radius * math.cos(angle), r * radius * math.sin(angle), z))
    for ring in range(2):
        for k in range(sides):
            a, b = ring * sides + k, ring * sides + (k + 1) % sides
            streak_faces.append((a, b, b + sides, a + sides))
    streak = mesh_object("RainStreakSource", streak_verts, streak_faces, get_collection("_Sources"))

    material = bpy.data.materials.new("RainDrop")
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    transparent = nb.new("ShaderNodeBsdfTransparent")
    body = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.25)
    body.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1.0)
    nb.wire(nb.shader_mix(0.42, transparent.outputs[0], body.outputs[0]), out.inputs["Surface"])
    streak.data.materials.append(material)

    group = bpy.data.node_groups.new("RainGN", "GeometryNodeTree")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    output = nodes.new("NodeGroupOutput")
    width, height, count = 44.0, 18.0, 46000

    def rand(low, high, seed):
        node = nodes.new("FunctionNodeRandomValue")
        node.data_type = "FLOAT"
        node.inputs["Min"].default_value = low
        node.inputs["Max"].default_value = high
        node.inputs["Seed"].default_value = seed
        return node.outputs[0]

    def math_node(op, *values):
        node = nodes.new("ShaderNodeMath")
        node.operation = op
        for i, value in enumerate(values):
            if isinstance(value, bpy.types.NodeSocket):
                links.new(value, node.inputs[i])
            else:
                node.inputs[i].default_value = value
        return node.outputs[0]

    points = nodes.new("GeometryNodePoints")
    points.inputs["Count"].default_value = count
    seconds = nodes.new("GeometryNodeInputSceneTime").outputs["Seconds"]
    camera = nodes.new("GeometryNodeInputActiveCamera")
    camera_info = nodes.new("GeometryNodeObjectInfo")
    camera_info.transform_space = "ORIGINAL"
    links.new(camera.outputs[0], camera_info.inputs["Object"])
    camera_xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(camera_info.outputs["Location"], camera_xyz.inputs[0])

    fall = math_node("MULTIPLY", seconds, rand(7.5, 9.8, 3))
    drift = math_node("MULTIPLY", seconds, 0.9)
    x = math_node("WRAP", math_node("ADD", rand(0, 1000, 1), drift),
                  math_node("ADD", camera_xyz.outputs[0], width / 2), math_node("SUBTRACT", camera_xyz.outputs[0], width / 2))
    y = math_node("WRAP", rand(0, 1000, 2),
                  math_node("ADD", camera_xyz.outputs[1], width / 2), math_node("SUBTRACT", camera_xyz.outputs[1], width / 2))
    z = math_node("WRAP", math_node("SUBTRACT", rand(0, 1000, 4), fall),
                  math_node("ADD", camera_xyz.outputs[2], height * 0.7), math_node("SUBTRACT", camera_xyz.outputs[2], height * 0.3))
    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(x, combine.inputs[0])
    links.new(y, combine.inputs[1])
    links.new(z, combine.inputs[2])
    links.new(combine.outputs[0], points.inputs["Position"])

    instance = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(points.outputs[0], instance.inputs["Points"])
    source = nodes.new("GeometryNodeObjectInfo")
    source.transform_space = "ORIGINAL"
    source.inputs["Object"].default_value = streak
    source.inputs["As Instance"].default_value = True
    links.new(source.outputs["Geometry"], instance.inputs["Instance"])
    links.new(rand(0.65, 1.5, 6), instance.inputs["Scale"])
    links.new(instance.outputs[0], output.inputs[0])

    rain = add_object("Rain", bpy.data.meshes.new("Rain"), coll)
    rain.modifiers.new("Rain", "NODES").node_group = group
    rain.visible_shadow = False
    rain.visible_volume_scatter = False
    rain.cycles.use_motion_blur = False
    return rain


def build_fog(coll):
    verts = [(x, y, z) for x in (-320, 520) for y in (-680, 600) for z in (-6, 70)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    fog = mesh_object("Fog", verts, faces, coll, smooth=False)
    material = bpy.data.materials.new("Fog")
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    volume = nb.new("ShaderNodeVolumePrincipled", Anisotropy=0.45)
    volume.inputs["Color"].default_value = (0.82, 0.88, 0.95, 1.0)
    volume.inputs["Emission Color"].default_value = (0.74, 0.79, 0.84, 1.0)
    nb.wire(volume.outputs[0], out.inputs["Volume"])
    fog.data.materials.append(material)
    return fog, volume


def build_world():
    world = bpy.data.worlds.new("LastSeenSky")
    bpy.context.scene.world = world
    nb = Nodes(world.node_tree)
    out = nb.new("ShaderNodeOutputWorld")
    coords = nb.new("ShaderNodeTexCoord").outputs["Generated"]
    night = nb.new("ShaderNodeTexEnvironment", image=load_image(ASSETS / "polyhaven" / "hdri" / f"{POLYHAVEN_HDRIS['night']}.hdr"))
    morning = nb.new("ShaderNodeTexEnvironment", image=load_image(ASSETS / "polyhaven" / "hdri" / f"{POLYHAVEN_HDRIS['morning']}.hdr"))
    nb.wire(coords, night.inputs["Vector"])
    nb.wire(coords, morning.inputs["Vector"])
    night_bg = nb.new("ShaderNodeBackground", Strength=0.35)
    nb.wire(night.outputs["Color"], night_bg.inputs["Color"])
    morning_bg = nb.new("ShaderNodeBackground", Strength=1.0)
    nb.wire(morning.outputs["Color"], morning_bg.inputs["Color"])
    blend = nb.new("ShaderNodeMixShader")
    nb.wire(night_bg.outputs[0], blend.inputs[1])
    nb.wire(morning_bg.outputs[0], blend.inputs[2])
    nb.wire(blend.outputs[0], out.inputs["Surface"])
    return blend.inputs[0], night_bg.inputs["Strength"]


# ---------------------------------------------------------------------------
# The BMW
# ---------------------------------------------------------------------------
CAR_SCALE = 0.745          # the demo model is modelled larger than life
CAR_DROP = {"Floor", "Light", "1M", "carRigArrow", "carRigOutline", "carController"}
# Resting positions the demo file gets from its rig, which is not imported.
CAR_SHELL_POSITION = (0.623, 0.0, 0.581)
CAR_WHEELS = {  # wheel: (brake, hub position in file coordinates, brake y, is front)
    "Tire1": ("Brake1", (-1.382, 1.06, 0.451), 1.06, True),
    "Tire2": ("Brake2", (-1.382, -1.06, 0.451), -1.06, True),
    "Tire3": ("Brake3", (2.24, -1.023, 0.452), -1.043, False),
    "Tire4": ("Brake4", (2.24, 1.023, 0.452), 1.043, False),
}


def shell_to_root(point):
    """Convert a point from the BMW file's coordinates to car coordinates (+Y forward, +X right)."""
    x, y, z = point
    return Vector((y * CAR_SCALE, -(x - 0.5) * CAR_SCALE, z * CAR_SCALE))


def paint_material(material):
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Metallic=0.55, Roughness=0.34)
    bsdf.inputs["Base Color"].default_value = (0.016, 0.02, 0.027, 1.0)
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.035
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    coords = nb.new("ShaderNodeTexCoord").outputs["Object"]
    big, _ = drop_height(nb, coords, 60.0, 0.34)
    small, _ = drop_height(nb, coords, 150.0, 0.30)
    beads = nb.math("ADD", big, nb.math("MULTIPLY", small, 0.45))
    bump = nb.new("ShaderNodeBump", Strength=1.0, Distance=0.03)
    nb.wire(beads, bump.inputs["Height"])
    nb.wire(bump.outputs[0], bsdf.inputs["Coat Normal"])


WIPER_PERIOD = 1.25
WIPER_SWEEP = 1.85
# Where each wiper pivots, in the BMW file's coordinates (x along the car, y across it).
WIPER_PIVOTS = ((-0.86, 0.0), (-0.86, -0.72))
SEAT_X = 0.37
MIRROR_LOCATION = (0.0, 0.34, 1.17)


def wipe_keep(nb, parts, clock, pivot):
    """1 where this blade has not wiped, 0 just behind it, then filling back in as rain lands."""
    u = nb.math("SUBTRACT", parts.outputs[1], pivot[1])
    w = nb.math("MULTIPLY", nb.math("SUBTRACT", parts.outputs[0], pivot[0]), 1.16)
    angle = nb.math("ARCTAN2", w, u)
    reach = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", u, u), nb.math("MULTIPLY", w, w)))
    inside = nb.math("MULTIPLY", nb.math("GREATER_THAN", reach, 0.25),
                     nb.math("MULTIPLY", nb.math("LESS_THAN", reach, 0.98),
                             nb.math("MULTIPLY", nb.math("GREATER_THAN", angle, -0.08), nb.math("LESS_THAN", angle, WIPER_SWEEP + 0.08))))
    tau = nb.math("FRACT", nb.math("DIVIDE", clock, WIPER_PERIOD))
    position = nb.map_range(angle, 0.0, WIPER_SWEEP)
    tau0 = nb.math("DIVIDE", nb.math("ARCCOSINE", nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", position, 2.0))), math.tau)
    late = nb.math("GREATER_THAN", tau, nb.math("SUBTRACT", 1.0, tau0))
    middle = nb.math("GREATER_THAN", tau, tau0)
    age = nb.mix(middle, nb.math("ADD", tau, tau0), nb.math("SUBTRACT", tau, tau0))
    age = nb.mix(late, age, nb.math("SUBTRACT", tau, nb.math("SUBTRACT", 1.0, tau0)))
    filled = nb.math("MINIMUM", nb.math("MULTIPLY", age, 2.4), 1.0)
    return nb.mix(inside, (1, 1, 1, 1), filled)


def glass_material(material, car):
    """Windows with real refracting water beads; the wipers clear them from the windshield."""
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    texture_coords = nb.new("ShaderNodeTexCoord")
    texture_coords.object = car.body
    coords = texture_coords.outputs["Object"]
    big_height, big_mask = drop_height(nb, coords, 42.0, 0.40)
    small_height, small_mask = drop_height(nb, coords, 105.0, 0.36)
    parts = nb.new("ShaderNodeSeparateXYZ")
    nb.wire(coords, parts.inputs[0])
    clock = nb.new("ShaderNodeValue", label="WipeClock")
    car.wipe_clock = clock
    keep = nb.math("MINIMUM", wipe_keep(nb, parts, clock.outputs[0], WIPER_PIVOTS[0]),
                   wipe_keep(nb, parts, clock.outputs[0], WIPER_PIVOTS[1]))
    height = nb.math("MULTIPLY", nb.math("ADD", big_height, nb.math("MULTIPLY", small_height, 0.5)), keep)
    mask = nb.math("MULTIPLY", nb.math("MAXIMUM", big_mask, small_mask), keep)
    bump = nb.new("ShaderNodeBump", Strength=1.0, Distance=0.02)
    nb.wire(height, bump.inputs["Height"])
    clear = nb.new("ShaderNodeBsdfTransparent")
    clear.inputs["Color"].default_value = (0.9, 0.92, 0.91, 1.0)
    gloss = nb.new("ShaderNodeBsdfGlossy", Roughness=0.03)
    nb.wire(bump.outputs[0], gloss.inputs["Normal"])
    # A fixed reflectivity: a Fresnel node would reflect everything when seen from inside the car.
    pane = nb.shader_mix(0.07, clear.outputs[0], gloss.outputs[0])
    # Thin-wall transmission: a one-sided pane has no back face for a true refraction ray to leave through.
    bead = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.03, IOR=1.33)
    bead.inputs["Transmission Weight"].default_value = 1.0
    bead.inputs["Thin Wall"].default_value = True
    bead.inputs["Specular IOR Level"].default_value = 1.0
    nb.wire(bump.outputs[0], bead.inputs["Normal"])
    nb.wire(nb.shader_mix(mask, pane, bead.outputs[0]), out.inputs["Surface"])


def leather_material(name, color, roughness=0.42):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Roughness=roughness)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    coords = nb.new("ShaderNodeTexCoord").outputs["Object"]
    grain = nb.new("ShaderNodeTexVoronoi", Scale=420.0, feature="DISTANCE_TO_EDGE")
    nb.wire(coords, grain.inputs["Vector"])
    bump = nb.new("ShaderNodeBump", Strength=0.35, Distance=0.002)
    nb.wire(grain.outputs[0], bump.inputs["Height"])
    nb.wire(bump.outputs[0], bsdf.inputs["Normal"])
    return material


def unseen(light_object, in_reflections=False):
    """Blender draws lamps as glowing shapes; these should only cast their light."""
    light_object.visible_camera = False
    light_object.visible_glossy = in_reflections
    return light_object


def spot_light(name, parent, location, target, coll, energy, size_deg, blend, temperature=4300.0):
    light = bpy.data.lights.new(name, "SPOT")
    light.energy = energy
    light.spot_size = math.radians(size_deg)
    light.spot_blend = blend
    light.shadow_soft_size = 0.04
    light.use_temperature = True
    light.temperature = temperature
    obj = add_object(name, light, coll)
    obj.parent = parent
    obj.location = location
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = look_at(location, target)
    return unseen(obj, in_reflections=True)


class Car:
    pass


def build_car(layout, coll):
    source = ASSETS / "car_bmw27" / "bmw27" / "bmw27_gpu.blend"
    with bpy.data.libraries.load(str(source), link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n not in CAR_DROP and not n.startswith("Camera")]
        dst.collections = ["Tire", "DiskBrake"]
    instanced = set()
    for name in ("Tire", "DiskBrake"):
        instanced |= {o.name for o in bpy.data.collections[name].objects}

    car = Car()
    car.root = add_object("CAR_ROOT", None, coll)
    car.body = add_object("CAR_BODY", None, coll)
    car.body.parent = car.root
    car.body.scale = (CAR_SCALE, CAR_SCALE, CAR_SCALE)
    car.body.rotation_euler = (0.0, 0.0, -math.pi / 2)
    car.body.location = (0.0, 0.5 * CAR_SCALE, 0.03)

    objects = {o.name: o for o in dst.objects if o is not None}
    for obj in objects.values():
        obj.animation_data_clear()
        for constraint in list(obj.constraints):
            obj.constraints.remove(constraint)
        if obj.name not in instanced:
            coll.objects.link(obj)
    shell = objects["carShell"]
    shell.parent = car.body
    shell.matrix_parent_inverse = Matrix.Identity(4)
    shell.location = CAR_SHELL_POSITION
    shell.rotation_euler = (0.0, 0.0, 0.0)
    car.shell = shell
    car.objects = objects

    # Materials
    paint_material(bpy.data.materials["CarShellNew"])
    glass_material(bpy.data.materials["Glass"], car)
    objects["Interior"].data.materials[0] = leather_material("CabinLeather", (0.028, 0.026, 0.024))

    # Wheels: Steer (Z) > Spin (Y) > original wheel instance, so they roll and turn correctly.
    car.spins, car.steers = [], []
    for wheel, (brake, hub, brake_y, front) in CAR_WHEELS.items():
        tire, disc = objects[wheel], objects[brake]
        if tire.scale.y < 0:
            # The file mirrors this wheel with a negative scale, which turns its face inward.
            tire.scale.y = disc.scale.y = -tire.scale.y
            tire.rotation_euler.z = disc.rotation_euler.z = math.pi
        steer = add_object(f"{wheel}_steer", None, coll)
        steer.parent = shell
        steer.location = Vector(hub) - Vector(CAR_SHELL_POSITION)
        spin = add_object(f"{wheel}_spin", None, coll)
        spin.parent = steer
        tire.parent = spin
        tire.matrix_parent_inverse = Matrix.Identity(4)
        tire.location = (0.0, 0.0, 0.0)
        disc.parent = steer
        disc.matrix_parent_inverse = Matrix.Identity(4)
        disc.location = (0.0, brake_y - hub[1], 0.0)
        car.spins.append(spin)
        if front:
            car.steers.append(steer)

    # Headlights. The left one is the lamp he cracked a year ago: dimmer and aimed a little off.
    target_y = 40.0
    car.head_left = spot_light("HeadlightCracked", car.root, (-0.56, 2.05, 0.66), (-1.6, target_y, -1.6), coll, 520.0, 62.0, 0.75)
    car.head_right = spot_light("HeadlightGood", car.root, (0.56, 2.05, 0.66), (0.9, target_y, -0.7), coll, 850.0, 58.0, 0.75)
    car.tail_lamps = []
    for side, x in (("L", -0.58), ("R", 0.58)):
        lamp = bpy.data.lights.new(f"Tail{side}", "POINT")
        lamp.energy = 2.2
        lamp.color = (1.0, 0.05, 0.03)
        lamp.shadow_soft_size = 0.06
        obj = add_object(f"Tail{side}", lamp, coll)
        obj.parent = car.root
        obj.location = (x, -2.32, 0.86)
        car.tail_lamps.append(unseen(obj, in_reflections=True))
    return car


def animate_car(car, layout):
    wheel_radius = 0.35
    rolled = 0.0
    root = car.root
    root.rotation_mode = "XYZ"
    for frame in range(1, LAST_FRAME + 2):
        position, tangent, speed = layout.car_frame(frame)
        heading = math.atan2(-tangent.x, tangent.y)
        rolled += speed / FPS / wheel_radius
        if frame % 2 == 1 or frame == LAST_FRAME + 1:
            root.location = position
            root.keyframe_insert("location", frame=frame)
            root.rotation_euler = (0.0, 0.0, heading)
            root.keyframe_insert("rotation_euler", frame=frame, index=2)
            for spin in car.spins:
                # Forward travel is -X in the file's axes, so wheels turn about -Y.
                spin.rotation_euler = (0.0, -rolled, 0.0)
                spin.keyframe_insert("rotation_euler", frame=frame, index=1)


# ---------------------------------------------------------------------------
# Images the script paints itself: posters, road sign, dashboard, phone, plate
# ---------------------------------------------------------------------------
FONT_FILES = {"impact": r"C:\Windows\Fonts\impact.ttf", "bold": r"C:\Windows\Fonts\arialbd.ttf", "regular": r"C:\Windows\Fonts\arial.ttf"}
_font_ids = {}


def font_id(kind):
    if kind not in _font_ids:
        path = FONT_FILES[kind]
        _font_ids[kind] = blf.load(path) if os.path.exists(path) else 0
    return _font_ids[kind]


def read_image(path):
    """Pixels as a float array, rows bottom-up, RGB in display space."""
    image = bpy.data.images.load(str(path))
    width, height = image.size
    channels = image.channels
    data = np.empty(width * height * channels, dtype=np.float32)
    image.pixels.foreach_get(data)
    bpy.data.images.remove(image)
    return data.reshape(height, width, channels)[..., :3]


def resize_image(array, width, height):
    ys = np.linspace(0, array.shape[0] - 1, height)
    xs = np.linspace(0, array.shape[1] - 1, width)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, array.shape[0] - 1), np.minimum(x0 + 1, array.shape[1] - 1)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = array[y0][:, x0] * (1 - fx) + array[y0][:, x1] * fx
    bottom = array[y1][:, x0] * (1 - fx) + array[y1][:, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def blotches(height, width, cells, rng):
    """Smooth random pattern in 0..1, used for paper stains and grime."""
    coarse = rng.random((cells, max(int(cells * width / height), 2), 1)).astype(np.float32)
    return resize_image(coarse, width, height)[..., 0]


class Canvas:
    """Origin at bottom-left, sizes in pixels, colours in display space."""

    def __init__(self, width, height, color=(1.0, 1.0, 1.0)):
        self.w, self.h = width, height
        self.rgb = np.ones((height, width, 3), dtype=np.float32) * np.array(color, dtype=np.float32)

    def rect(self, x0, y0, x1, y1, color):
        self.rgb[int(y0):int(y1), int(x0):int(x1)] = color

    def paste(self, array, x, y, width, height):
        self.rgb[int(y):int(y) + height, int(x):int(x) + width] = resize_image(array, width, height)

    def text(self, string, x, y, size, color=(0.0, 0.0, 0.0), font="bold", align="left", alpha=1.0):
        fid = font_id(font)
        blf.size(fid, size)
        width = blf.dimensions(fid, string)[0]
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        buffer = imbuf.new((self.w, self.h))
        with blf.bind_imbuf(fid, buffer):
            blf.color(fid, 1.0, 1.0, 1.0, 1.0)
            blf.position(fid, x, y, 0)
            blf.draw_buffer(fid, string)
        GENERATED.mkdir(parents=True, exist_ok=True)
        temp = GENERATED / "_text.png"
        imbuf.write(buffer, filepath=str(temp))
        buffer.free()
        mask = read_image(temp)[..., 0:1] * alpha
        temp.unlink()
        self.rgb = self.rgb * (1 - mask) + np.array(color, dtype=np.float32) * mask

    def ring(self, cx, cy, radius, thickness, color):
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        distance = np.hypot(xx - cx, yy - cy)
        mask = np.clip(1.0 - np.abs(distance - radius) / thickness, 0, 1)[..., None]
        self.rgb = self.rgb * (1 - mask) + np.array(color, dtype=np.float32) * mask

    def line(self, x0, y0, x1, y1, thickness, color):
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
        distance = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
        mask = np.clip(1.0 - distance / thickness, 0, 1)[..., None]
        self.rgb = self.rgb * (1 - mask) + np.array(color, dtype=np.float32) * mask

    def save(self, name, alpha=None):
        GENERATED.mkdir(parents=True, exist_ok=True)
        path = GENERATED / f"{name}.png"
        image = bpy.data.images.new(f"_{name}", self.w, self.h, alpha=alpha is not None)
        fourth = np.ones((self.h, self.w, 1), dtype=np.float32) if alpha is None else alpha[..., None]
        image.pixels.foreach_set(np.concatenate([np.clip(self.rgb, 0, 1), fourth], axis=2).astype(np.float32).ravel())
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        bpy.data.images.remove(image)
        return path


def paper_poster(key, photo, name_line, seen_line, date_line, faded=0.0, seed=0):
    rng = np.random.default_rng(seed + 100)
    canvas = Canvas(900, 1270, (0.93, 0.91, 0.84))
    canvas.text("MISSING", 450, 1038, 215, (0.70, 0.04, 0.04), font="impact", align="center")
    canvas.rect(120, 322, 780, 1012, (0.05, 0.05, 0.05))
    canvas.paste(photo, 130, 332, 640, 670)
    canvas.text(name_line, 450, 178, 118, (0.05, 0.05, 0.05), font="impact", align="center")
    canvas.text(seen_line, 450, 112, 50, (0.08, 0.08, 0.08), align="center")
    canvas.text(date_line, 450, 50, 50, (0.08, 0.08, 0.08), align="center")
    grime = blotches(1270, 900, 26, rng)[..., None]
    canvas.rgb *= 0.86 + 0.14 * grime
    if faded > 0.0:
        washed = blotches(1270, 900, 9, rng)[..., None]
        paper = np.array((0.74, 0.70, 0.60), dtype=np.float32)
        canvas.rgb = canvas.rgb * (1 - faded) + (paper * (0.75 + 0.25 * washed)) * faded
        streaks = resize_image(rng.random((1, 70, 1)).astype(np.float32), 900, 1270)
        canvas.rgb *= 1.0 - 0.28 * faded * streaks
    return canvas.save(f"poster_{key}")


def photo_or_placeholder(name):
    path = GENERATED / f"photo_{name}.png"
    if path.exists():
        return read_image(path)
    return np.ones((8, 8, 3), dtype=np.float32) * np.array((0.32, 0.34, 0.38), dtype=np.float32)


POSTER_SPECS = {
    "old": ("school", "ANJALI, 9", "Last seen: Old Forest Road", "6 Nov 2025", 0.62),
    "smile": ("school", "ANJALI, 9", "Last seen: Old Forest Road", "6 Nov 2025", 0.0),
    "crying": ("crying", "ANJALI, 9", "Last seen: Old Forest Road", "6 Nov 2025", 0.0),
    "blood": ("blood", "ANJALI, 9", "Last seen: Old Forest Road", "6 Nov 2025", 0.0),
    "road": ("road", "ANJALI, 9", "Last seen: Old Forest Road", "6 Nov 2025", 0.0),
    "digging": ("digging", "ANJALI, 9", "Last seen: Old Forest Road", "6 Nov 2025", 0.08),
    "prasath_a": ("prasath_a", "PRASATH, 38", "Last seen: Old Forest Road", "6 Nov 2026", 0.0),
    "prasath_b": ("prasath_b", "PRASATH, 38", "Last seen: Old Forest Road", "6 Nov 2026", 0.0),
}


def paint_posters():
    paths = {}
    for seed, (key, (photo, name, seen, date, faded)) in enumerate(POSTER_SPECS.items()):
        paths[key] = paper_poster(key, photo_or_placeholder(photo), name, seen, date, faded, seed)
    return paths


def paint_detour_sign():
    canvas = Canvas(1400, 700, (0.86, 0.66, 0.05))
    black = (0.03, 0.03, 0.03)
    for box in ((0, 0, 1400, 26), (0, 674, 1400, 700), (0, 0, 26, 700), (1374, 0, 1400, 700)):
        canvas.rect(*box, black)
    canvas.text("DETOUR", 700, 410, 215, black, font="impact", align="center")
    canvas.text("OLD FOREST ROAD", 590, 170, 112, black, font="impact", align="center")
    canvas.rect(1090, 196, 1270, 244, black)
    for i in range(70):
        canvas.rect(1270 + i, 220 - (70 - i), 1271 + i, 220 + (70 - i), black)
    rng = np.random.default_rng(5)
    canvas.rgb *= (0.8 + 0.2 * blotches(700, 1400, 18, rng))[..., None]
    return canvas.save("sign_detour")


def paint_dashboard():
    canvas = Canvas(1100, 440, (0.006, 0.007, 0.01))
    glow = (0.95, 0.62, 0.34)
    for cx, label in ((225, "km/h"), (875, "x1000 rpm")):
        canvas.ring(cx, 215, 182, 3.0, glow)
        steps = 11
        for i in range(steps):
            angle = math.radians(225 - 270 * i / (steps - 1))
            canvas.line(cx + 158 * math.cos(angle), 215 + 158 * math.sin(angle),
                        cx + 178 * math.cos(angle), 215 + 178 * math.sin(angle), 3.0, glow)
            if i % 2 == 0:
                value = str(i * 20) if cx == 225 else str(i)
                canvas.text(value, cx + 128 * math.cos(angle), 203 + 128 * math.sin(angle), 26, glow, align="center")
        canvas.text(label, cx, 92, 22, glow, align="center")
    needle = (0.95, 0.16, 0.08)
    canvas.line(225, 215, 225 + 135 * math.cos(math.radians(135)), 215 + 135 * math.sin(math.radians(135)), 4.0, needle)
    canvas.line(875, 215, 875 + 135 * math.cos(math.radians(150)), 215 + 135 * math.sin(math.radians(150)), 4.0, needle)
    canvas.rect(435, 110, 665, 330, (0.012, 0.014, 0.02))
    canvas.text("6 NOV", 550, 240, 62, (1.0, 0.86, 0.66), font="bold", align="center")
    canvas.text("23:47", 550, 168, 46, glow, font="bold", align="center")
    canvas.text("17 C  RAIN", 550, 124, 24, glow, align="center")
    return canvas.save("dash_cluster")


def paint_radio():
    canvas = Canvas(420, 130, (0.004, 0.005, 0.008))
    canvas.text("FM 98.3", 210, 42, 58, (0.35, 0.75, 1.0), font="bold", align="center")
    return canvas.save("radio_display")


def paint_phone():
    canvas = Canvas(360, 760, (0.02, 0.025, 0.035))
    canvas.text("23:58", 180, 560, 92, (0.92, 0.94, 1.0), align="center")
    canvas.text("6 November", 180, 516, 26, (0.75, 0.78, 0.85), align="center")
    canvas.text("No Service", 180, 340, 40, (1.0, 0.35, 0.3), align="center")
    canvas.text("Emergency calls", 180, 282, 26, (0.7, 0.72, 0.78), align="center")
    canvas.text("unavailable", 180, 248, 26, (0.7, 0.72, 0.78), align="center")
    return canvas.save("phone_screen")


def paint_plate():
    canvas = Canvas(520, 120, (0.92, 0.92, 0.88))
    canvas.rect(0, 0, 520, 5, (0.05, 0.05, 0.05))
    canvas.rect(0, 115, 520, 120, (0.05, 0.05, 0.05))
    canvas.text("KA 05 MR 4417", 260, 28, 70, (0.03, 0.03, 0.03), font="bold", align="center")
    return canvas.save("number_plate")


def paint_handprint():
    """A small child's muddy hand on glass, with an alpha channel."""
    size = 512
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    def blob(cx, cy, rx, ry, angle):
        c, s = math.cos(angle), math.sin(angle)
        u = ((xx - cx) * c + (yy - cy) * s) / rx
        v = (-(xx - cx) * s + (yy - cy) * c) / ry
        return u * u + v * v

    field = blob(256, 190, 92, 100, 0.0)
    for cx, cy, rx, ry, angle in ((168, 338, 25, 84, 0.25), (226, 372, 26, 100, 0.08), (286, 376, 26, 100, -0.05),
                                  (343, 340, 24, 84, -0.2), (390, 192, 28, 84, -0.82)):
        field = np.minimum(field, blob(cx, cy, rx, ry, angle))
    rng = np.random.default_rng(3)
    grain = blotches(size, size, 90, rng)
    alpha = np.clip((1.05 - field) * 3.0, 0, 1) * (0.55 + 0.6 * grain)
    for x in rng.integers(180, 340, 7):   # muddy water running down the glass
        bottom = int(rng.integers(10, 80))
        alpha[bottom:120, x - 3:x + 3] = np.maximum(alpha[bottom:120, x - 3:x + 3], 0.55 * grain[bottom:120, x - 3:x + 3])
    canvas = Canvas(size, size, (0.075, 0.05, 0.03))
    canvas.rgb *= (0.7 + 0.6 * grain)[..., None]
    return canvas.save("handprint", alpha=np.clip(alpha, 0, 1).astype(np.float32))


# ---------------------------------------------------------------------------
# Inside the BMW: the demo model only has rough seats, so the cockpit details are added here
# ---------------------------------------------------------------------------
def rounded_box(name, size, material, coll, parent=None, location=(0, 0, 0), rotation=(0, 0, 0), radius=0.01):
    x, y, z = size[0] / 2, size[1] / 2, size[2] / 2
    verts = [(sx * x, sy * y, sz * z) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    obj = mesh_object(name, verts, faces, coll)
    bevel = obj.modifiers.new("Round", "BEVEL")
    bevel.width = radius
    bevel.segments = 4
    bevel.limit_method = "NONE"
    obj.data.materials.append(material)
    obj.parent = parent
    obj.location = location
    obj.rotation_euler = rotation
    return obj


def torus_mesh(name, major, minor, coll, segments=64, sides=14):
    verts, faces = [], []
    for i in range(segments):
        a = math.tau * i / segments
        for j in range(sides):
            b = math.tau * j / sides
            r = major + minor * math.cos(b)
            verts.append((r * math.cos(a), r * math.sin(a), minor * math.sin(b)))
    for i in range(segments):
        for j in range(sides):
            faces.append((i * sides + j, ((i + 1) % segments) * sides + j,
                          ((i + 1) % segments) * sides + (j + 1) % sides, i * sides + (j + 1) % sides))
    return mesh_object(name, verts, faces, coll)


def emission_image_material(name, path, strength):
    material = bpy.data.materials.new(name)
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    picture = nb.new("ShaderNodeTexImage", image=load_image(path))
    glow = nb.new("ShaderNodeEmission", Strength=strength)
    nb.wire(picture.outputs["Color"], glow.inputs["Color"])
    nb.wire(glow.outputs[0], out.inputs["Surface"])
    return material, glow.inputs["Strength"]


def image_plane(name, width, height, material, coll, parent=None, location=(0, 0, 0), rotation=(0, 0, 0)):
    x, y = width / 2, height / 2
    obj = mesh_object(name, [(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)], [(0, 1, 2, 3)], coll, smooth=False)
    layer = obj.data.uv_layers.new(name="UVMap")
    for loop, uv in zip(layer.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop.uv = uv
    obj.data.materials.append(material)
    obj.parent = parent
    obj.location = location
    obj.rotation_euler = rotation
    return obj


def build_wipers(car, coll):
    rubber = leather_material("WiperRubber", (0.008, 0.008, 0.008), 0.5)
    car.wiper_sweeps = []
    # Local axes on the glass: X across the car, Y up the windshield, Z out of it.
    slope = math.atan2(0.60, 1.07)
    x_axis = Vector((1, 0, 0))
    y_axis = Vector((0, -math.cos(slope), math.sin(slope)))
    z_axis = x_axis.cross(y_axis)
    orientation = Matrix((x_axis, y_axis, z_axis)).transposed().to_quaternion()
    for index, (px, py) in enumerate(WIPER_PIVOTS):
        pivot = add_object(f"WiperPivot{index}", None, coll)
        pivot.parent = car.root
        pivot.location = shell_to_root((px, py, 1.245)) + Vector((0, 0, 0.016))
        pivot.rotation_mode = "QUATERNION"
        pivot.rotation_quaternion = orientation
        sweep = add_object(f"WiperSweep{index}", None, coll)
        sweep.parent = pivot
        length = 0.72
        rounded_box(f"WiperArm{index}", (0.34, 0.016, 0.010), rubber, coll, sweep, (0.17, 0, 0.012), radius=0.003)
        rounded_box(f"WiperBlade{index}", (length - 0.2, 0.013, 0.016), rubber, coll, sweep, (0.2 + (length - 0.2) / 2, 0, 0.004), radius=0.003)
        car.wiper_sweeps.append(sweep)


def open_up_dashboard(car):
    """The demo's cabin is a solid shell; cut away the part in front of the seats so the dash can be seen."""
    interior = car.objects["Interior"]
    mesh = interior.data
    offset = Vector(CAR_SHELL_POSITION) + Vector((0.0, 0.0, -0.6)) + interior.location
    edit = bmesh.new()
    edit.from_mesh(mesh)
    doomed = []
    for face in edit.faces:
        centre = Vector((face.calc_center_median().x * interior.scale.x, face.calc_center_median().y * interior.scale.y,
                         face.calc_center_median().z * interior.scale.z)) + offset
        if centre.x < -0.10 and centre.z > 0.70:
            doomed.append(face)
    bmesh.ops.delete(edit, geom=doomed, context="FACES")
    edit.to_mesh(mesh)
    edit.free()
    log(f"cabin faces removed in front of the seats: {len(doomed)}")


def build_cockpit(car, images, coll):
    open_up_dashboard(car)
    leather = leather_material("WheelLeather", (0.012, 0.012, 0.013), 0.46)
    plastic = leather_material("DashPlastic", (0.016, 0.016, 0.017), 0.55)
    root = car.root
    seat_x = SEAT_X

    # Steering wheel on the right: this is an Indian-registered, right-hand-drive car.
    wheel_mount = add_object("WheelMount", None, coll)
    wheel_mount.parent = root
    wheel_mount.location = (seat_x, 0.47, 0.83)
    wheel_mount.rotation_euler = (math.radians(90 - 24), 0, 0)
    wheel_spin = add_object("WheelSpin", None, coll)
    wheel_spin.parent = wheel_mount
    rim = torus_mesh("WheelRim", 0.178, 0.017, coll)
    rim.data.materials.append(leather)
    rim.parent = wheel_spin
    rounded_box("WheelHub", (0.13, 0.095, 0.05), leather, coll, wheel_spin, (0, 0, 0.012), radius=0.02)
    for label, angle in (("L", 0.0), ("R", math.pi), ("B", -math.pi / 2)):
        reach = 0.12
        rounded_box(f"WheelSpoke{label}", (reach, 0.045, 0.016), plastic, coll, wheel_spin,
                    (reach * 0.62 * math.cos(angle), reach * 0.62 * math.sin(angle), 0.004), (0, 0, angle), radius=0.006)
    car.wheel_spin = wheel_spin

    # Instrument hood, cluster and radio
    rounded_box("ClusterHood", (0.40, 0.17, 0.05), plastic, coll, root, (seat_x, 0.70, 0.985), (math.radians(-6), 0, 0), radius=0.02)
    rounded_box("ClusterBack", (0.38, 0.03, 0.17), plastic, coll, root, (seat_x, 0.775, 0.885), (math.radians(-12), 0, 0), radius=0.01)
    cluster_material, car.cluster_glow = emission_image_material("ClusterImage", images["dash"], 1.4)
    image_plane("ClusterDisplay", 0.34, 0.136, cluster_material, coll, root, (seat_x, 0.755, 0.885),
                (math.radians(78), 0, 0))
    radio_material, car.radio_glow = emission_image_material("RadioImage", images["radio"], 1.2)
    image_plane("RadioDisplay", 0.13, 0.04, radio_material, coll, root, (-0.02, 0.72, 0.80),
                (math.radians(78), 0, 0))

    # Soft dash glow that lights his chest and face from below
    glow = bpy.data.lights.new("DashGlow", "AREA")
    glow.size = 0.32
    glow.color = (1.0, 0.72, 0.45)
    glow.energy = 1.1
    car.dash_light = unseen(add_object("DashGlow", glow, coll))
    car.dash_light.parent = root
    car.dash_light.location = (seat_x, 0.70, 0.84)
    car.dash_light.rotation_mode = "QUATERNION"
    car.dash_light.rotation_quaternion = look_at((seat_x, 0.70, 0.84), (seat_x, 0.0, 1.10))

    # Rear-view mirror
    mirror_aim = Vector((0.38, -0.92, -0.16)).normalized()
    housing = rounded_box("MirrorHousing", (0.25, 0.022, 0.075), plastic, coll, root, MIRROR_LOCATION, radius=0.012)
    housing.rotation_mode = "QUATERNION"
    housing.rotation_quaternion = mirror_aim.to_track_quat("-Y", "Z")
    silver = bpy.data.materials.new("MirrorSilver")
    nb = Nodes(silver.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Metallic=1.0, Roughness=0.015)
    bsdf.inputs["Base Color"].default_value = (0.75, 0.76, 0.78, 1.0)
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    rounded_box("MirrorGlass", (0.235, 0.002, 0.062), silver, coll, housing, (0, -0.0118, 0), radius=0.0009)

    # Cold breath on the back seat, seen in the mirror
    fog_verts = [(x, y, z) for x in (-0.45, 0.45) for y in (-0.42, 0.42) for z in (0.68, 1.26)]
    fog_faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    breath = mesh_object("BackSeatBreath", fog_verts, fog_faces, coll, smooth=False)
    breath.parent = root
    breath.location = (0.05, -0.92, 0.0)
    material = bpy.data.materials.new("Breath")
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    position = nb.new("ShaderNodeTexCoord").outputs["Object"]
    moved = nb.new("ShaderNodeMapping", Scale=(6.0, 6.0, 3.0))
    nb.wire(position, moved.inputs["Vector"])
    moved.inputs["Location"].driver_add("default_value", 2).driver.expression = f"-frame/{FPS}*0.55"
    puffs = nb.new("ShaderNodeTexNoise", Detail=5.0, Roughness=0.6)
    nb.wire(moved.outputs[0], puffs.inputs["Vector"])
    shape = nb.map_range(puffs.outputs[0], 0.52, 0.72, smooth=True)
    amount = nb.new("ShaderNodeValue")
    amount.outputs[0].default_value = 0.0
    car.breath_amount = amount
    volume = nb.new("ShaderNodeVolumePrincipled", Anisotropy=0.2)
    volume.inputs["Color"].default_value = (0.9, 0.95, 1.0, 1.0)
    nb.wire(nb.math("MULTIPLY", shape, amount.outputs[0]), volume.inputs["Density"])
    nb.wire(volume.outputs[0], out.inputs["Volume"])
    breath.data.materials.append(material)
    back_light = bpy.data.lights.new("BackSeatChill", "POINT")
    back_light.color = (0.55, 0.72, 1.0)
    back_light.shadow_soft_size = 0.2
    back_light.energy = 0.0
    car.back_light = unseen(add_object("BackSeatChill", back_light, coll))
    car.back_light.parent = root
    car.back_light.location = (0.1, -0.5, 1.15)

    # Silver tape over the cracked left headlight
    tape_material = bpy.data.materials.new("GafferTape")
    nb = Nodes(tape_material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.38, Metallic=0.35)
    bsdf.inputs["Base Color"].default_value = (0.52, 0.52, 0.5, 1.0)
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    lens = car.objects["lightsGlass"]
    for index, (dy, dz, angle, length) in enumerate(((0.0, 0.015, 0.45, 0.30), (0.03, -0.015, -0.5, 0.26), (-0.03, -0.01, 0.1, 0.22))):
        columns = 18
        verts, faces = [], []
        for i in range(columns + 1):
            t = (i / columns - 0.5) * length
            for across in (-0.022, 0.022):
                verts.append((t, across, 0.0))
        for i in range(columns):
            faces.append((i * 2, i * 2 + 1, i * 2 + 3, i * 2 + 2))
        strip = mesh_object(f"HeadlightTape{index}", verts, faces, coll)
        strip.data.materials.append(tape_material)
        strip.parent = root
        strip.location = (-0.585 + dy, 2.06, 0.70 + dz)
        strip.rotation_euler = (math.radians(90) - 0.45, angle, 0.0)
        wrap = strip.modifiers.new("OnLens", "SHRINKWRAP")
        wrap.target = lens
        wrap.wrap_method = "NEAREST_SURFACEPOINT"
        wrap.offset = 0.004

    # Number plate
    plate = bpy.data.materials.get("Plate")
    if plate:
        for node in plate.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                node.image = load_image(images["plate"])


def cut_open_door(car, coll):
    """A second copy of the body with the driver's door as its own hinged part, for after he gets out."""
    shell = car.shell
    body_mesh = shell.data.copy()
    # Apply the mirror by hand so one side of the car can be edited.
    editable = bmesh.new()
    editable.from_mesh(body_mesh)
    geometry = editable.verts[:] + editable.edges[:] + editable.faces[:]
    bmesh.ops.mirror(editable, geom=geometry, axis="Y", merge_dist=0.002)
    editable.to_mesh(body_mesh)
    editable.free()
    offset = Vector(CAR_SHELL_POSITION)

    def is_door(face):
        centre = face.calc_center_median() + offset
        return centre.y > 0.55 and -0.57 < centre.x < 1.15 and 0.5 < centre.z < 1.31

    door_part = bmesh.new()
    door_part.from_mesh(body_mesh)
    bmesh.ops.delete(door_part, geom=[f for f in door_part.faces if not is_door(f)], context="FACES")
    door_mesh = bpy.data.meshes.new("carDoor")
    door_part.to_mesh(door_mesh)
    door_part.free()
    body_part = bmesh.new()
    body_part.from_mesh(body_mesh)
    bmesh.ops.delete(body_part, geom=[f for f in body_part.faces if is_door(f)], context="FACES_ONLY")
    body_part.to_mesh(body_mesh)
    body_part.free()
    body_mesh.shade_smooth()
    door_mesh.materials.append(shell.data.materials[0])
    door_mesh.shade_smooth()

    cut = add_object("carShell_doorOpen", body_mesh, coll)
    cut.parent = car.body
    cut.location = shell.location
    smooth = cut.modifiers.new("Smooth", "SUBSURF")
    smooth.levels = smooth.render_levels = 2

    hinge_position = Vector((-0.55, 1.18, 0.9))
    hinge = add_object("DoorHinge", None, coll)
    hinge.parent = car.body
    hinge.location = hinge_position
    hinge.rotation_euler = (0.0, 0.0, math.radians(58))
    door = add_object("carDoor", door_mesh, coll)
    door.parent = hinge
    door.location = offset - hinge_position
    smooth = door.modifiers.new("Smooth", "SUBSURF")
    smooth.levels = smooth.render_levels = 2
    door.modifiers.new("Thick", "SOLIDIFY").thickness = 0.03
    car.shell_open = cut
    car.door = door

    window = car.objects["Window"]
    window_mesh = window.data.copy()
    edit = bmesh.new()
    edit.from_mesh(window_mesh)
    bmesh.ops.mirror(edit, geom=edit.verts[:] + edit.edges[:] + edit.faces[:], axis="Y", merge_dist=0.002)
    scale = window.scale.x
    def driver_side(face):
        centre = face.calc_center_median() * scale
        file_x = centre.x + window.location.x + CAR_SHELL_POSITION[0]
        return centre.y > 0.72 and file_x < 1.25
    bmesh.ops.delete(edit, geom=[f for f in edit.faces if driver_side(f)], context="FACES")
    edit.to_mesh(window_mesh)
    edit.free()
    car.window_open = add_object("Window_doorOpen", window_mesh, coll)
    car.window_open.parent = window.parent
    car.window_open.matrix_parent_inverse = window.matrix_parent_inverse
    car.window_open.location = window.location
    car.window_open.scale = window.scale
    smooth = car.window_open.modifiers.new("Smooth", "SUBSURF")
    smooth.levels = smooth.render_levels = 2


# ---------------------------------------------------------------------------
# People: Rocketbox avatars driven by Rocketbox motion capture
# ---------------------------------------------------------------------------
ADULT_HIP = 0.8952
FEMALE_CLIP_HIP = 0.919


class ClipLibrary:
    """
    Rocketbox animation files carry no bind pose, so their keys are relative to each file's own
    rest pose and cannot simply be assigned to an avatar. Each clip is kept on its own armature in
    a private scene and sampled bone by bone when an avatar needs it.
    """

    def __init__(self):
        self.scene = bpy.data.scenes.new("_ClipBake")
        self.armatures = {}

    def armature(self, name):
        if name in self.armatures:
            return self.armatures[name]
        before_objects, before_actions = set(bpy.data.objects), set(bpy.data.actions)
        bpy.ops.import_scene.fbx(filepath=str(ASSETS / "rocketbox" / "_anim" / f"{name}.fbx"), use_anim=True)
        new_objects = [o for o in bpy.data.objects if o not in before_objects]
        arm = next(o for o in new_objects if o.type == "ARMATURE")
        for obj in new_objects:
            if obj is not arm:
                bpy.data.objects.remove(obj)
        for action in [a for a in bpy.data.actions if a not in before_actions]:
            if arm.animation_data and arm.animation_data.action is action:
                continue
            bpy.data.actions.remove(action)
        for owner in list(arm.users_collection):
            owner.objects.unlink(arm)
        self.scene.collection.objects.link(arm)
        self.armatures[name] = arm
        return arm

    def length(self, name):
        return int(self.armature(name).animation_data.action.frame_range[1])

    def sample(self, name, frame):
        arm = self.armature(name)
        self.scene.frame_set(frame)
        evaluated = arm.evaluated_get(self.scene.view_layers[0].depsgraph)
        bones = {bone.name: bone.matrix.copy() for bone in evaluated.pose.bones}
        return bones, evaluated.location.copy(), evaluated.rotation_euler.copy(), evaluated.scale.copy()

    def cleanup(self):
        for arm in self.armatures.values():
            action = arm.animation_data.action if arm.animation_data else None
            data = arm.data
            bpy.data.objects.remove(arm)
            bpy.data.armatures.remove(data)
            if action:
                bpy.data.actions.remove(action)
        bpy.data.scenes.remove(self.scene)


class Actor:
    def __init__(self, label, avatar, coll, clips):
        self.label = label
        self.clips = clips
        before = set(bpy.data.objects)
        bpy.ops.import_scene.fbx(filepath=str(ASSETS / "rocketbox" / avatar / f"{avatar}.fbx"), use_anim=False)
        new_objects = [o for o in bpy.data.objects if o not in before]
        self.arm = next(o for o in new_objects if o.type == "ARMATURE")
        self.mesh = next(o for o in new_objects if o.type == "MESH")
        for obj in new_objects:
            if obj.type == "EMPTY":
                bpy.data.objects.remove(obj)
        for obj in (self.arm, self.mesh):
            move_to(obj, coll)
        for bone in self.arm.data.bones:
            if bone.name.startswith("Bip02"):   # children use a different prefix
                bone.name = bone.name.replace("Bip02", "Bip01")
        self.rest_hip = float(self.arm.location.z)
        self.root = add_object(f"{label}_ROOT", None, coll)
        self.arm.parent = self.root
        self.arm.name = f"{label}_Arm"
        self.mesh.name = f"{label}_Body"
        self.mesh.data.shade_smooth()
        subdivide = self.mesh.modifiers.new("Smooth", "SUBSURF")
        subdivide.levels = 1
        subdivide.render_levels = 2
        self.root.rotation_mode = "XYZ"
        self.baked = {}
        self.arm.animation_data_create()
        self.improve_materials()

    def improve_materials(self):
        for material in self.mesh.data.materials:
            nodes = material.node_tree.nodes
            bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue
            bsdf.inputs["Metallic"].default_value = 0.0
            bsdf.inputs["Roughness"].default_value = 0.42
            for node in nodes:
                if node.type == "TEX_IMAGE" and node.image and any(k in node.image.filepath for k in ("_normal", "_specular")):
                    node.image.colorspace_settings.name = "Non-Color"
            if "_head" in material.name:
                bsdf.inputs["Subsurface Weight"].default_value = 0.35
                bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.18)
                bsdf.inputs["Subsurface Scale"].default_value = 0.012

    def bake(self, clip, first=1, count=1, tweaks=None, in_place=True, key=None):
        """
        Build an action for this avatar from a stretch of a clip.
        tweaks = {bone name: Euler in that bone's own axes}, added on top of the motion capture
        so that children follow (lean the spine and the head and arms come with it).
        """
        key = key or f"{clip}:{first}:{count}"
        if key in self.baked:
            return self.baked[key]
        tweaks = tweaks or {}
        reference_hip = FEMALE_CLIP_HIP if clip.startswith("f_") else ADULT_HIP
        hip_ratio = self.rest_hip / reference_hip
        bones = self.arm.data.bones
        rest_relative = {}
        for bone in bones:
            rest_relative[bone.name] = (bone.parent.matrix_local.inverted() @ bone.matrix_local) if bone.parent else bone.matrix_local.copy()
        samples = [self.clips.sample(clip, first + i) for i in range(count)]
        origin = samples[0][1].copy()
        action = bpy.data.actions.new(f"{self.label}|{key}")
        action.use_fake_user = True
        slot = action.slots.new("OBJECT", self.label)
        layer = action.layers.new("Base")
        strip = layer.strips.new(type="KEYFRAME")
        bag = strip.channelbag(slot, ensure=True)

        def write(path, index, values):
            curve = bag.fcurves.new(path, index=index)
            curve.keyframe_points.add(len(values))
            coords = np.empty(len(values) * 2, dtype=np.float32)
            coords[0::2] = np.arange(1, len(values) + 1, dtype=np.float32)
            coords[1::2] = values
            curve.keyframe_points.foreach_set("co", coords)
            curve.keyframe_points.foreach_set("interpolation", [1] * len(values))  # LINEAR
            curve.update()

        for bone in bones:
            frames = []
            for matrices, _, _, _ in samples:
                source = matrices.get(bone.name)
                if source is None:
                    frames.append((1.0, 0.0, 0.0, 0.0))
                    continue
                relative = (matrices[bone.parent.name].inverted() @ source) if bone.parent else source
                quat = rest_relative[bone.name].to_quaternion().inverted() @ relative.to_quaternion()
                if bone.name in tweaks:
                    quat = quat @ tweaks[bone.name].to_quaternion()
                frames.append(tuple(quat))
            frames = np.array(frames, dtype=np.float32)
            # Keep neighbouring frames on the same side of the quaternion sphere.
            for i in range(1, len(frames)):
                if np.dot(frames[i], frames[i - 1]) < 0.0:
                    frames[i] = -frames[i]
            path = f'pose.bones["{bone.name}"].rotation_quaternion'
            for index in range(4):
                write(path, index, frames[:, index])

        for index in range(3):
            write("location", index, np.array([(s[1][index] - origin[index]) if (in_place and index < 2) else (s[1][index] * (hip_ratio if index == 2 else 1.0)) for s in samples], dtype=np.float32))
            write("rotation_euler", index, np.array([s[2][index] for s in samples], dtype=np.float32))
            write("scale", index, np.array([s[3][index] for s in samples], dtype=np.float32))
        if in_place:
            for index in range(2):
                curve = bag.fcurves.find("location", index=index)
                curve.keyframe_points.foreach_set("co", np.column_stack([np.arange(1, count + 1), np.zeros(count)]).astype(np.float32).ravel())
                curve.update()
        stride = (samples[-1][1] - samples[0][1]).length * hip_ratio
        self.baked[key] = (action, slot, count, stride)
        return self.baked[key]

    def hold(self, baked, first, last, speed=1.0, track_name=None):
        """Play a baked action between two film frames, looping if it is shorter."""
        action, slot, count, _ = baked
        data = self.arm.animation_data
        track = data.nla_tracks.new()
        track.name = track_name or action.name[:40]
        strip = track.strips.new(action.name[:60], int(first), action)
        strip.action_slot = slot
        strip.extrapolation = "NOTHING"
        strip.use_auto_blend = False
        strip.blend_in = strip.blend_out = 0.0
        strip.action_frame_start = 1.0
        strip.action_frame_end = float(max(count, 2))
        strip.scale = 1.0 / speed
        span = max(count - 1, 1) / speed
        strip.repeat = max((last - first + 1) / span, 1.0 / max(span, 1.0))
        return strip

    def overlay(self, baked, first, last, blend=6):
        """Play a variation over whatever is underneath, easing in and out (a head turn, a flinch)."""
        strip = self.hold(baked, first, last)
        strip.blend_in = strip.blend_out = float(blend)
        return strip

    def add_arm_ik(self, side, target):
        bone = self.arm.pose.bones[f"Bip01 {side} Forearm"]
        constraint = bone.constraints.new("IK")
        constraint.target = target
        constraint.chain_count = 2
        constraint.use_rotation = False
        return constraint

    def at(self, frame, location, heading=0.0, kind="CONSTANT"):
        """Place the character. heading 0 means facing -Y, which is how the avatars were built."""
        key_values(self.root, "location", [(frame, Vector(location))], kind=kind)
        key_values(self.root, "rotation_euler", [(frame, Euler((0.0, 0.0, heading)))], kind=kind)


# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------
def configure_render(scene):
    scene.render.engine = "CYCLES"
    scene.render.fps = FPS
    scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.frame_start, scene.frame_end = 1, LAST_FRAME
    cycles = scene.cycles
    cycles.samples = FINAL_SAMPLES
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.04
    cycles.adaptive_min_samples = 16
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cycles.max_bounces = 4
    cycles.diffuse_bounces = 1
    cycles.glossy_bounces = 2
    cycles.transmission_bounces = 4
    cycles.volume_bounces = 0
    cycles.transparent_max_bounces = 8
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 4.0
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.texture_limit_render = "2048"
    # Skip forest instances far outside the view: big speed gain, invisible at night in fog.
    scene.render.use_simplify = True
    scene.render.simplify_subdivision_render = 6
    cycles.use_camera_cull = True
    cycles.camera_cull_margin = 0.15
    cycles.use_distance_cull = True
    cycles.distance_cull_margin = 45.0
    scene.render.compositor_device = "CPU"   # no dependence on the machine's graphics driver
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    scene.render.use_persistent_data = True
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 40


# ---------------------------------------------------------------------------
# Set pieces
# ---------------------------------------------------------------------------
PHOTO_FRAME = 5000   # frames past the end of the film, used to pose people for the poster photographs
BOOTH = Vector((3400.0, 0.0, 0.0))   # where the school portrait is taken, outside the forest and the fog


def heading_to(direction):
    """Root rotation that makes an avatar (built facing -Y) face a direction."""
    return math.atan2(direction.x, -direction.y)


def copy_of(source, name, coll, location, scale=1.0, rotation=(0.0, 0.0, 0.0)):
    obj = add_object(name, source.data, coll)
    obj.location = location
    obj.scale = (scale, scale, scale)
    obj.rotation_euler = rotation
    return obj


def surface_tree(obj):
    mesh = obj.data
    return BVHTree.FromPolygons([v.co.copy() for v in mesh.vertices], [p.vertices[:] for p in mesh.polygons])


def surface_point(obj, tree, origin, direction):
    """Point just in front of the trunk across the poster's whole area, centred on the trunk."""
    inverse = obj.matrix_world.inverted()
    side = Vector((-direction.y, direction.x, 0.0)).normalized()
    local_direction = (inverse.to_3x3() @ direction).normalized()
    hits = []
    for column in range(-4, 5):
        for row in range(-4, 5):
            dx, dz = column * 0.06, row * 0.08
            start = origin + side * dx + Vector((0, 0, dz))
            hit = tree.ray_cast(inverse @ start, local_direction)
            if hit[0] is not None:
                hits.append((dx, ((obj.matrix_world @ hit[0]) - start).dot(direction)))
    if not hits:
        return None
    centre = sum(h[0] for h in hits) / len(hits)
    return origin + side * centre + direction * min(h[1] for h in hits)


def poster_material(key, path):
    material = bpy.data.materials.new(f"Poster_{key}")
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.3)
    picture = nb.new("ShaderNodeTexImage", image=load_image(path))
    nb.wire(picture.outputs["Color"], bsdf.inputs["Base Color"])
    wrinkles = nb.new("ShaderNodeTexNoise", Scale=22.0, Detail=6.0)
    nb.wire(nb.new("ShaderNodeTexCoord").outputs["Object"], wrinkles.inputs["Vector"])
    bump = nb.new("ShaderNodeBump", Strength=0.3, Distance=0.01)
    nb.wire(wrinkles.outputs[0], bump.inputs["Height"])
    nb.wire(bump.outputs[0], bsdf.inputs["Normal"])
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    return material


def poster_object(name, material, position, normal, coll, flap=True, seed=0):
    width, height, cols, rows = 0.45, 0.635, 10, 14
    verts, faces, uvs, weights = [], [], [], []
    for j in range(rows + 1):
        for i in range(cols + 1):
            u, v = i / cols, j / rows
            x = (u - 0.5) * width
            verts.append((x, (v - 0.5) * height, -1.1 * x * x))
            uvs.append((u, v))
            weights.append((1.0 - v) ** 1.5)
    for j in range(rows):
        for i in range(cols):
            a = j * (cols + 1) + i
            faces.append((a, a + 1, a + cols + 2, a + cols + 1))
    obj = mesh_object(name, verts, faces, coll)
    layer = obj.data.uv_layers.new(name="UVMap")
    for polygon in obj.data.polygons:
        for loop_index in polygon.loop_indices:
            layer.data[loop_index].uv = uvs[obj.data.loops[loop_index].vertex_index]
    obj.data.materials.append(material)
    obj.location = position
    z_axis = normal.normalized()
    x_axis = Vector((0, 0, 1)).cross(z_axis).normalized()
    y_axis = z_axis.cross(x_axis)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Matrix((x_axis, y_axis, z_axis)).transposed().to_quaternion()
    if flap:
        group = obj.vertex_groups.new(name="Flap")
        for index, weight in enumerate(weights):
            group.add([index], weight, "REPLACE")
        wave = obj.modifiers.new("Flap", "WAVE")
        wave.use_x, wave.use_y = True, True
        wave.use_normal = True
        wave.height = 0.035
        wave.width = 0.32
        wave.speed = 0.09
        wave.narrowness = 1.2
        wave.time_offset = -seed * 7.0
        wave.vertex_group = "Flap"
    return obj


def tube_curve(name, polylines, radius, material, coll, parent=None):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    for points in polylines:
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for point, co in zip(spline.points, points):
            point.co = (*co, 1.0)
    curve.materials.append(material)
    obj = add_object(name, curve, coll)
    obj.parent = parent
    return obj


def build_bicycle(coll):
    """A child's bicycle, seen only for seconds at night. Thin tubes and spoked wheels hold up well in that light."""
    root = add_object("Bicycle", None, coll)
    root.rotation_mode = "XYZ"
    frame = leather_material("BikeFrame", (0.20, 0.03, 0.03), 0.3)
    metal = leather_material("BikeMetal", (0.45, 0.45, 0.46), 0.25)
    rubber = leather_material("BikeRubber", (0.008, 0.008, 0.008), 0.55)
    radius, front, rear = 0.27, 0.42, -0.42
    hub_z = radius + 0.017
    for wheel_name, y in (("BikeWheelFront", front), ("BikeWheelRear", rear)):
        tyre = torus_mesh(wheel_name, radius, 0.017, coll, segments=48, sides=8)
        tyre.data.materials.append(rubber)
        tyre.parent = root
        tyre.location = (0, y, hub_z)
        tyre.rotation_euler = (0, math.pi / 2, 0)
        spokes = []
        for i in range(16):
            angle = math.tau * i / 16
            spokes.append([(0, y, hub_z), (0, y + radius * math.cos(angle), hub_z + radius * math.sin(angle))])
        tube_curve(f"{wheel_name}Spokes", spokes, 0.0012, metal, coll, root)
    tube_curve("BikeFrameTubes", [
        [(0, rear, hub_z), (0, -0.10, hub_z - 0.03), (0, -0.16, 0.62), (0, rear, hub_z)],
        [(0, -0.10, hub_z - 0.03), (0, 0.24, 0.64), (0, -0.16, 0.60)],
        [(0, 0.24, 0.64), (0, 0.27, 0.74), (0, front, hub_z)],
    ], 0.014, frame, coll, root)
    tube_curve("BikeBars", [[(0, 0.27, 0.74), (0, 0.25, 0.82)], [(-0.22, 0.22, 0.84), (0, 0.25, 0.82), (0.22, 0.22, 0.84)]], 0.011, metal, coll, root)
    seat = rounded_box("BikeSeat", (0.11, 0.21, 0.04), rubber, coll, root, (0, -0.18, 0.70), radius=0.018)
    seat.rotation_euler = (0.06, 0, 0)
    tube_curve("BikeSeatPost", [[(0, -0.16, 0.60), (0, -0.18, 0.69)]], 0.012, metal, coll, root)
    tube_curve("BikeCrank", [[(0, -0.10, hub_z - 0.03), (0.05, -0.02, hub_z - 0.12)], [(0, -0.10, hub_z - 0.03), (-0.05, -0.18, hub_z + 0.06)]], 0.007, metal, coll, root)
    lamp_glass = bpy.data.materials.new("BikeLampGlass")
    nb = Nodes(lamp_glass.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    glow = nb.new("ShaderNodeEmission", Strength=60.0)
    glow.inputs["Color"].default_value = (1.0, 0.93, 0.8, 1.0)
    nb.wire(glow.outputs[0], out.inputs["Surface"])
    rounded_box("BikeLamp", (0.04, 0.03, 0.04), lamp_glass, coll, root, (0, 0.31, 0.74), radius=0.012)
    for child in root.children:
        key_visible(child, [(fr(FLASHBACK[0]), fr(85) - 1)])
    return root


def build_mud_grave(position, coll):
    rings, sectors, reach = 18, 56, 1.15
    verts = [(0.0, 0.0, 0.19)]
    flat = [(0.0, 0.0, 0.035)]
    for ring in range(1, rings + 1):
        r = reach * ring / rings
        for sector in range(sectors):
            angle = math.tau * sector / sectors
            x, y = r * math.cos(angle), r * math.sin(angle)
            lump = 0.05 * mnoise.noise(Vector((x * 3.1, y * 3.1, 0.3))) + 0.018 * mnoise.noise(Vector((x * 9.0, y * 9.0, 1.3)))
            falloff = 1.0 - (r / reach) ** 2
            verts.append((x, y, 0.19 * falloff + lump * falloff ** 0.5 - 0.015))
            flat.append((x, y, 0.035 * falloff + 0.5 * lump * falloff ** 0.5 - 0.015))
    faces = [(0, 1 + s, 1 + (s + 1) % sectors) for s in range(sectors)]
    for ring in range(rings - 1):
        for s in range(sectors):
            a = 1 + ring * sectors + s
            b = 1 + ring * sectors + (s + 1) % sectors
            faces.append((a, a + sectors, b + sectors, b))
    obj = mesh_object("MudGrave", verts, faces, coll)
    obj.location = position
    obj.shape_key_add(name="Basis")
    sunk = obj.shape_key_add(name="Sunk")
    for point, co in zip(sunk.data, flat):
        point.co = co
    material = bpy.data.materials.new("WetMud")
    nb = Nodes(material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled")
    tile = nb.new("ShaderNodeMapping", Scale=(1 / 1.6, 1 / 1.6, 1 / 1.6))
    nb.wire(nb.new("ShaderNodeTexCoord").outputs["Object"], tile.inputs["Vector"])
    color, rough, normal = pbr_nodes(nb, "mud_forest", tile.outputs[0])
    nb.wire(nb.mix(0.72, color, (0.15, 0.075, 0.035, 1.0)), bsdf.inputs["Base Color"])
    nb.wire(nb.math("ADD", nb.math("MULTIPLY", rough, 0.22), 0.04), bsdf.inputs["Roughness"])
    normal_map = nb.new("ShaderNodeNormalMap", Strength=1.6)
    nb.wire(normal, normal_map.inputs["Color"])
    nb.wire(normal_map.outputs[0], bsdf.inputs["Normal"])
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(material)
    return obj, sunk


def extract_shoe(actor, position, coll):
    """Her own left shoe, cut from the avatar, so the shoe in the mud matches the girl."""
    source = actor.mesh
    groups = {source.vertex_groups[name].index for name in ("Bip01 L Foot", "Bip01 L Toe0") if name in source.vertex_groups}
    mesh = source.data.copy()
    edit = bmesh.new()
    edit.from_mesh(mesh)
    layer = edit.verts.layers.deform.verify()
    drop = [v for v in edit.verts if sum(w for g, w in v[layer].items() if g in groups) < 0.5]
    bmesh.ops.delete(edit, geom=drop, context="VERTS")
    centre = sum((v.co for v in edit.verts), Vector()) / max(len(edit.verts), 1)
    for v in edit.verts:
        v.co -= centre
    edit.to_mesh(mesh)
    edit.free()
    shoe = add_object("SchoolShoe", mesh, coll)
    smooth = shoe.modifiers.new("Smooth", "SUBSURF")
    smooth.levels = smooth.render_levels = 2
    shoe.scale = (0.014, 0.014, 0.014)
    shoe.location = position
    shoe.rotation_euler = (0.9, 0.2, 1.1)
    return shoe


def ghostify(material, value_nodes):
    """Blend from the living girl to the drowned one with a keyframed value (mud, pallor, wet cloth)."""
    tree = material.node_tree
    bsdf = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    base_socket = bsdf.inputs["Base Color"]
    if not base_socket.is_linked:
        return
    source = base_socket.links[0].from_socket
    nb = Nodes(tree, clear=False)
    ghost = nb.new("ShaderNodeValue")
    value_nodes.append(ghost)
    grey = nb.new("ShaderNodeRGBToBW")
    nb.wire(source, grey.inputs[0])
    is_head = "_head" in material.name
    pale = nb.mix(0.8 if is_head else 0.35, source, nb.mix(grey.outputs[0], (0.34, 0.37, 0.38, 1), (0.92, 0.96, 0.98, 1)))
    grime = nb.new("ShaderNodeTexNoise", Scale=0.26, Detail=7.0, Roughness=0.62)
    nb.wire(nb.new("ShaderNodeTexCoord").outputs["Object"], grime.inputs["Vector"])
    mud = nb.map_range(grime.outputs[0], 0.47 if not is_head else 0.54, 0.62, smooth=True)
    dirty = nb.mix(mud, pale, (0.035, 0.024, 0.014, 1))
    nb.wire(nb.mix(ghost.outputs[0], source, dirty), base_socket)
    roughness = nb.math("ADD", 0.42, nb.math("MULTIPLY", ghost.outputs[0], -0.1))
    nb.wire(roughness, bsdf.inputs["Roughness"])


def blacken_eyes(actor, value_nodes):
    """Her eyes keep their colour in photographs and turn solid black as a ghost."""
    source = actor.mesh
    eye_groups = {source.vertex_groups[n].index for n in ("Bip01 LEye", "Bip01 REye")}
    head = next(m for m in source.data.materials if "_head" in m.name)
    eye = head.copy()
    eye.name = "AnjaliEye"
    nb = Nodes(eye.node_tree, clear=False)
    out = next(n for n in eye.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
    alive = out.inputs["Surface"].links[0].from_socket
    black = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.06)
    black.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    ghost = nb.new("ShaderNodeValue")
    value_nodes.append(ghost)
    nb.wire(nb.shader_mix(ghost.outputs[0], alive, black.outputs[0]), out.inputs["Surface"])
    source.data.materials.append(eye)
    slot = len(source.data.materials) - 1
    weights = {}
    for vertex in source.data.vertices:
        weights[vertex.index] = sum(g.weight for g in vertex.groups if g.group in eye_groups)
    changed = 0
    for polygon in source.data.polygons:
        if all(weights[v] > 0.5 for v in polygon.vertices):
            polygon.material_index = slot
            changed += 1
    log(f"Anjali eye polygons: {changed}")


def make_khaki_uniform(actor):
    """Indian police wear khaki: recolour the blue uniform and remove its foreign shoulder patches."""
    material = next(m for m in actor.mesh.data.materials if "_body" in m.name)
    node = next(n for n in material.node_tree.nodes if n.type == "TEX_IMAGE" and "body_color" in n.image.filepath)
    target = GENERATED / "officer_body_khaki.png"
    if not target.exists():
        rgb = read_image(bpy.path.abspath(node.image.filepath))
        luminance = rgb @ np.array((0.3, 0.59, 0.11), dtype=np.float32)
        skin = (rgb[..., 0] > rgb[..., 2] * 1.2) & (rgb[..., 1] > rgb[..., 2] * 0.95) & (rgb[..., 0] > rgb[..., 1] * 1.08) & (luminance > 0.1)
        khaki = np.array((0.60, 0.50, 0.30), dtype=np.float32)
        reference = max(float(np.median(luminance[~skin])), 0.05)
        shaded = khaki * np.clip(luminance[..., None] / reference, 0.2, 1.5)
        canvas = Canvas(rgb.shape[1], rgb.shape[0])
        canvas.rgb = np.where(skin[..., None], rgb, shaded).astype(np.float32)
        canvas.save("officer_body_khaki")
    node.image = load_image(target)
    head = next(m for m in actor.mesh.data.materials if "_head" in m.name)
    node = next(n for n in head.node_tree.nodes if n.type == "TEX_IMAGE" and "head_color" in n.image.filepath)
    target = GENERATED / "officer_head_khaki.png"
    if not target.exists():
        rgb = read_image(bpy.path.abspath(node.image.filepath))
        blue = rgb[..., 2] > rgb[..., 0] * 1.04
        luminance = rgb @ np.array((0.3, 0.59, 0.11), dtype=np.float32)
        reference = max(float(np.median(luminance[blue])), 0.05) if blue.any() else 0.3
        shaded = np.array((0.60, 0.50, 0.30), dtype=np.float32) * np.clip(luminance[..., None] / reference, 0.2, 1.5)
        canvas = Canvas(rgb.shape[1], rgb.shape[0])
        canvas.rgb = np.where(blue[..., None], shaded, rgb).astype(np.float32)
        canvas.save("officer_head_khaki")
    node.image = load_image(target)


class Story:
    pass


def build_set(story):
    layout, models = story.layout, story.models
    coll = get_collection("Set")
    height = layout.height

    # The old tree that holds her grave, with exposed roots around its base.
    tree_source = models["island_tree_01"]["island_tree_01_LOD0"]
    big = layout.big_tree
    big_z = height(big.x, big.y)
    story.big_tree = copy_of(tree_source, "BigOldTree", coll, (big.x, big.y, big_z - 0.15), 5.4, (0.0, 0.0, 1.1))
    bpy.context.view_layer.update()
    copy_of(models["root_cluster_01"]["root_cluster_01"], "BigTreeRoots", coll, (big.x, big.y, big_z - 0.25), 1.35, (0, 0, 0.5))
    copy_of(models["pine_roots"]["pine_roots_a"], "BigTreeRootsB", coll, (big.x - 2.6, big.y + 0.6, big_z - 0.02), 1.5, (0, 0, 2.3))
    copy_of(models["pine_roots"]["pine_roots_b"], "BigTreeRootsC", coll, (big.x + 0.5, big.y - 2.4, big_z - 0.02), 1.5, (0, 0, 4.1))

    # The fallen tree and the road closure on the highway.
    fallen_source = models["island_tree_02"]["island_tree_02_LOD0"]
    copy_of(fallen_source, "FallenTree", coll, (-9.5, 118.0, 0.25), 4.2, (0.05, math.radians(87), math.radians(5)))
    copy_of(models["dead_tree_trunk_02"]["dead_tree_trunk_02"], "FallenTrunkPiece", coll, (-2.0, 112.0, 0.32), 1.7, (0.0, 0.1, 0.25))
    barrier_source = next(o for o in import_polyhaven_model("concrete_road_barrier", get_collection("_Sources")) if o.type == "MESH")
    barrier_source.location = (0, 0, 0)
    for index, x in enumerate((-3.2, -0.6, 2.0)):
        copy_of(barrier_source, f"Barrier{index}", coll, (x, 104.0, 0.03), 1.0, (0.0, 0.0, 0.04 * index))

    # Detour sign
    sign_material = bpy.data.materials.new("DetourSign")
    nb = Nodes(sign_material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    picture = nb.new("ShaderNodeTexImage", image=load_image(story.images["sign"]))
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.32)
    nb.wire(picture.outputs["Color"], bsdf.inputs["Base Color"])
    nb.wire(picture.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.05   # road signs are retro-reflective
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    sign_at = Vector((6.4, 88.0, 0.0))
    image_plane("DetourBoard", 2.4, 1.2, sign_material, coll, None, (sign_at.x, sign_at.y, 1.9), (math.radians(90), 0, math.radians(-8)))
    post = leather_material("SignPost", (0.26, 0.26, 0.27), 0.45)
    for index, offset in enumerate((-0.9, 0.9)):
        rounded_box(f"SignPost{index}", (0.06, 0.06, 2.2), post, coll, None, (sign_at.x + offset, sign_at.y + 0.04, 1.1), radius=0.008)

    # Posters on dead trunks along the forest road. Their photographs get worse as he drives on.
    snag_source = models["dead_tree_trunk_02"]["dead_tree_trunk_02"]
    low_end = min(v.co.x for v in snag_source.data.vertices)
    snag_tree = surface_tree(snag_source)
    materials = {key: poster_material(key, path) for key, path in story.images["posters"].items()}
    story.posters = []
    story.hero_posters = {}

    def poster_at(seconds, side, key, facing_road, offset, hero=None, flap=True):
        s = float(layout.car_s[fr(seconds)])
        centre = layout.car_path.pos(s)
        tangent, right = layout.car_path.tangent(s), layout.car_path.right(s)
        place = centre + right * side * (FOREST_HALF_WIDTH + offset)
        scale = 0.75 + 0.3 * ((len(story.posters) * 37 % 10) / 10)
        ground = height(place.x, place.y)
        snag = copy_of(snag_source, f"PosterTrunk{len(story.posters)}", coll, (place.x, place.y, ground - low_end * scale - 0.25), scale,
                       (0.0, math.radians(-90 + (len(story.posters) * 13 % 7) - 3), 0.0))
        bpy.context.view_layer.update()
        outward = ((right * -side) * facing_road + (tangent * -1.0) * (1.0 - facing_road)).normalized()
        origin = Vector((place.x, place.y, ground + 1.62)) + outward * 2.5
        point = surface_point(snag, snag_tree, origin, -outward)
        if point is None:
            point = Vector((place.x, place.y, ground + 1.62)) + outward * 0.5
        poster = poster_object(f"Poster{len(story.posters)}_{key}", materials[key], point + outward * 0.012, outward, coll, flap=flap, seed=len(story.posters))
        story.posters.append(poster)
        if hero:
            story.hero_posters[hero] = poster
        return poster

    poster_at(25.1, 1, "old", 0.45, 1.4, hero="old", flap=False)
    for index, seconds in enumerate(np.arange(26.8, 33.0, 0.5)):
        poster_at(float(seconds), 1 if index % 2 == 0 else -1, "smile", 0.42, 1.2 + (index * 3 % 5) * 0.22)
    poster_at(40.2, 1, "smile", 0.62, 1.15, hero="smile")
    poster_at(43.4, 1, "crying", 0.62, 1.15, hero="crying")
    poster_at(46.4, 1, "blood", 0.62, 1.15, hero="blood")
    poster_at(49.6, 1, "road", 0.62, 1.15, hero="road")
    for seconds in np.arange(38.6, 55.0, 1.45):
        key = "smile" if seconds < 42 else "crying" if seconds < 45.5 else "blood" if seconds < 48.8 else "road"
        poster_at(float(seconds), -1, key, 0.5, 1.3)

    # Last poster on the big tree: the photograph of him digging. Next morning it is replaced with his own.
    parked = layout.parked_pos
    toward_car = (Vector((parked.x, parked.y, 0)) - Vector((big.x, big.y, 0))).normalized()
    origin = Vector((big.x, big.y, big_z + 1.55)) + toward_car * 8.0
    bark_tree = surface_tree(story.big_tree)
    bark = surface_point(story.big_tree, bark_tree, origin, -toward_car) or Vector((big.x, big.y, big_z + 1.55)) + toward_car * 1.5
    story.toward_car = toward_car
    story.bark_point = bark
    story.poster_digging = poster_object("PosterDigging", materials["digging"], bark + toward_car * 0.02, toward_car, coll, flap=False)
    prasath_material = bpy.data.materials.new("PosterPrasathSwap")
    nb = Nodes(prasath_material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.3)
    image_a = nb.new("ShaderNodeTexImage", image=load_image(story.images["posters"]["prasath_a"]))
    image_b = nb.new("ShaderNodeTexImage", image=load_image(story.images["posters"]["prasath_b"]))
    snap = nb.new("ShaderNodeValue")
    story.poster_snap = snap
    nb.wire(nb.mix(snap.outputs[0], image_a.outputs["Color"], image_b.outputs["Color"]), bsdf.inputs["Base Color"])
    nb.wire(bsdf.outputs[0], out.inputs["Surface"])
    story.poster_prasath = poster_object("PosterPrasath", prasath_material, bark + toward_car * 0.02, toward_car, coll, flap=False)

    # The grave, off to the side of where he will stand.
    side = Vector((-toward_car.y, toward_car.x, 0.0))
    distance = max((bark - Vector((big.x, big.y, bark.z))).length, 0.4)
    story.stand = Vector((bark.x, bark.y, 0.0)) + toward_car * 0.85
    story.stand.z = height(story.stand.x, story.stand.y)
    grave_xy = Vector((bark.x, bark.y, 0.0)) + toward_car * 1.5 + side * 1.55
    story.grave_position = Vector((grave_xy.x, grave_xy.y, height(grave_xy.x, grave_xy.y) + 0.01))
    story.grave, story.grave_sunk = build_mud_grave(story.grave_position, coll)
    log(f"big tree bark sits {distance:.2f} m from its centre")

    # Spade and bicycle
    story.spade = next(o for o in import_polyhaven_model("rusted_spade_01", get_collection("Props")) if o.type == "MESH")
    story.spade.name = "Spade"
    story.bicycle = build_bicycle(get_collection("Props"))

    # Hand-print that slams onto his window
    print_material = bpy.data.materials.new("Handprint")
    nb = Nodes(print_material.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    picture = nb.new("ShaderNodeTexImage", image=load_image(story.images["handprint"]))
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.5)
    pale = nb.mix(0.8, picture.outputs["Color"], (0.5, 0.38, 0.26, 1.0))
    nb.wire(pale, bsdf.inputs["Base Color"])
    nb.wire(pale, bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.35
    transparent = nb.new("ShaderNodeBsdfTransparent")
    nb.wire(nb.shader_mix(picture.outputs["Alpha"], transparent.outputs[0], bsdf.outputs[0]), out.inputs["Surface"])
    story.handprint = image_plane("Handprint", 0.17, 0.17, print_material, get_collection("Props"), story.car.root,
                                  (0.752, 0.06, 1.095), (math.radians(90), math.radians(-7), math.radians(90)))

    # Phone he carries
    props = get_collection("Props")
    phone_screen, story.phone_glow = emission_image_material("PhoneScreen", story.images["phone"], 1.6)
    story.hand_target = add_object("V_HandR_Target", None, props)
    story.hand_target_left = add_object("V_HandL_Target", None, props)
    case = leather_material("PhoneCase", (0.01, 0.01, 0.012), 0.3)
    story.phone = rounded_box("Phone", (0.074, 0.150, 0.009), case, props, story.hand_target, (0, 0, 0), radius=0.004)
    story.phone.rotation_mode = "QUATERNION"
    screen = image_plane("PhoneDisplay", 0.066, 0.14, phone_screen, props, story.phone, (0, 0, 0.0052))
    story.phone_parts = (story.phone, screen)
    beam = bpy.data.lights.new("PhoneTorch", "SPOT")
    beam.energy = 0.0
    beam.spot_size = math.radians(78)
    beam.spot_blend = 0.7
    beam.shadow_soft_size = 0.01
    beam.color = (0.92, 0.96, 1.0)
    story.torch = unseen(add_object("PhoneTorch", beam, props), in_reflections=True)
    story.torch.rotation_mode = "QUATERNION"
    story.torch.parent = story.hand_target


# ---------------------------------------------------------------------------
# People through the film
# ---------------------------------------------------------------------------
BLINK_WIDE = {"Bip01 LEyeBlinkTop": Euler((0, 0, -0.16)), "Bip01 REyeBlinkTop": Euler((0, 0, -0.16)),
              "Bip01 LEyeBlinkBottom": Euler((0, 0, 0.06)), "Bip01 REyeBlinkBottom": Euler((0, 0, 0.06))}
RECLINE = {"Bip01 Spine": Euler((0, 0, -0.26)), "Bip01 Spine1": Euler((0, 0, -0.04)),
           "Bip01 Neck": Euler((0, 0, 0.12)), "Bip01 Head": Euler((0, 0, 0.13))}


def with_tweaks(*groups):
    merged = {}
    for group in groups:
        merged.update(group)
    return merged


def key_constant(owner, prop, keys):
    key_values(owner, prop, keys, kind="CONSTANT")


def car_local_to_world(layout, frame, local):
    position, tangent, _ = layout.car_frame(frame)
    right = Vector((tangent.y, -tangent.x, 0.0))
    return position + right * local[0] + tangent * local[1] + Vector((0, 0, local[2]))


def direct_people(story):
    layout = story.layout
    prasath, anjali, officer, second = story.prasath, story.anjali, story.officer, story.second_officer
    parked, tangent = layout.parked_pos, layout.car_path.tangent(layout.parked_s)
    right = layout.car_path.right(layout.parked_s)
    photo = PHOTO_FRAME
    toward_car = story.toward_car

    # ---- Prasath at the wheel ----
    sit = "m_sit_chair_idle_neutral_01"
    drive = prasath.bake(sit, first=1, count=240, tweaks=RECLINE, key="drive")
    glance = prasath.bake(sit, first=1, count=2, key="glance_right",
                         tweaks=with_tweaks(RECLINE, {"Bip01 Head": Euler((-0.85, 0, 0.08)), "Bip01 Neck": Euler((-0.25, 0, 0.1))}))
    down = prasath.bake(sit, first=1, count=2, key="look_down",
                       tweaks=with_tweaks(RECLINE, {"Bip01 Head": Euler((0.1, 0, 0.5)), "Bip01 Neck": Euler((0, 0, 0.2))}))
    mirror = prasath.bake(sit, first=1, count=2, key="look_mirror",
                         tweaks=with_tweaks(RECLINE, BLINK_WIDE, {"Bip01 Head": Euler((0.3, 0, 0.0))}))
    flinch = prasath.bake(sit, first=1, count=2, key="flinch",
                         tweaks=with_tweaks(RECLINE, BLINK_WIDE, {"Bip01 Head": Euler((-1.15, 0.1, -0.05)), "Bip01 Neck": Euler((-0.3, 0, 0.0)),
                                                                   "Bip01 MJaw": Euler((0, 0, 0.16)), "Bip01 Spine1": Euler((0, 0.14, -0.04))}))
    prasath.hold(drive, fr(0), fr(65) - 1)
    prasath.hold(drive, fr(FLASHBACK[0]), fr(82.2) - 1)
    prasath.overlay(glance, fr(26.9), fr(27.8))
    prasath.overlay(mirror, fr(51.2), fr(59.0))
    prasath.overlay(down, fr(59.4), fr(63.0))
    prasath.overlay(flinch, fr(63.1), fr(65) - 1, blend=2)
    prasath.overlay(down, fr(FLASHBACK[0]), fr(81.4))

    ik_right = prasath.add_arm_ik("R", story.hand_target)
    ik_left = prasath.add_arm_ik("L", story.hand_target_left)
    seat = (SEAT_X, -0.06, -0.03)
    wheel_right = (SEAT_X + 0.168, 0.45, 0.85)
    wheel_left = (SEAT_X - 0.168, 0.45, 0.85)
    phone_held = (SEAT_X - 0.05, 0.16, 0.98)
    driving = list(range(fr(0) - 1, fr(65), 2)) + list(range(fr(FLASHBACK[0]), fr(82.2), 2))
    for frame in driving:
        _, direction, _ = layout.car_frame(frame)
        prasath.root.location = car_local_to_world(layout, frame, seat)
        prasath.root.keyframe_insert("location", frame=frame)
        prasath.root.rotation_euler = (0.0, 0.0, math.atan2(-direction.x, direction.y) + math.pi)
        prasath.root.keyframe_insert("rotation_euler", frame=frame)
        seconds = (frame - 1) / FPS
        hand = phone_held if 59.3 < seconds < 63.0 or seconds >= FLASHBACK[0] else wheel_right
        story.hand_target.location = car_local_to_world(layout, frame, hand)
        story.hand_target.keyframe_insert("location", frame=frame)
        story.hand_target_left.location = car_local_to_world(layout, frame, wheel_left)
        story.hand_target_left.keyframe_insert("location", frame=frame)
    # He is scaled slightly so a 1.82 m avatar fits a low coupe's roofline when seated.
    key_constant(prasath.root, "scale", [(fr(0), Vector((0.94, 0.94, 0.94))), (fr(65), Vector((1, 1, 1))), (fr(FLASHBACK[0]), Vector((0.94, 0.94, 0.94))),
                                        (fr(82.2), Vector((1, 1, 1)))])

    # ---- Prasath out of the car ----
    walk = prasath.bake("m_walk_slow_01", first=1, count=prasath.clips.length("m_walk_slow_01"), key="walk")
    speed = 1.35
    story.walk_speed = walk[3] / max(walk[2] - 1, 1) * FPS * speed
    door_position = Vector((parked.x, parked.y, 0.0)) + right * 1.25
    stand = story.stand
    walk_vector = Vector((stand.x - door_position.x, stand.y - door_position.y, 0.0))
    walk_distance = walk_vector.length
    direction = walk_vector.normalized()
    t_start, t_end = 65.0, 75.0
    story.walk_start, story.walk_direction, story.walk_distance = door_position, direction, walk_distance

    for frame in range(fr(t_start), fr(t_end), 4):
        seconds = (frame - 1) / FPS
        place = walker_position(story, seconds)
        prasath.root.location = place
        prasath.root.keyframe_insert("location", frame=frame)
        prasath.root.rotation_euler = (0.0, 0.0, heading_to(direction))
        prasath.root.keyframe_insert("rotation_euler", frame=frame)
        wobble = 0.02 * math.sin(frame * 0.23)
        hand_at = place + direction * 0.40 + Vector((-direction.y, direction.x, 0)) * -0.20 + Vector((0, 0, 1.13 + wobble))
        story.hand_target.location = hand_at
        story.hand_target.keyframe_insert("location", frame=frame)
    prasath.hold(walk, fr(t_start), fr(t_end) - 1, speed=speed)

    # ---- At the tree ----
    afraid = prasath.bake("m_idle_nervous_01", first=1, count=160, tweaks=BLINK_WIDE, key="afraid")
    terrified = prasath.bake("m_idle_nervous_01", first=1, count=2, key="terrified",
                            tweaks=with_tweaks(BLINK_WIDE, {"Bip01 MJaw": Euler((0, 0, 0.3)), "Bip01 Head": Euler((0, 0, -0.08))}))
    looking_down = prasath.bake("m_idle_nervous_01", first=1, count=2, key="look_grave",
                               tweaks=with_tweaks(BLINK_WIDE, {"Bip01 Head": Euler((0.0, 0, 0.5)), "Bip01 Neck": Euler((0, 0, 0.3)), "Bip01 Spine2": Euler((0, 0, 0.12))}))
    prasath.hold(afraid, fr(75), fr(80) - 1)
    prasath.hold(looking_down, fr(88), fr(92) - 1)
    prasath.hold(terrified, fr(92), fr(94))
    prasath.at(fr(75), stand, heading_to(-toward_car))
    to_grave = Vector((story.grave_position.x - stand.x, story.grave_position.y - stand.y, 0.0)).normalized()
    prasath.at(fr(88), stand, heading_to(to_grave))
    prasath.at(fr(92), stand + toward_car * 0.15, heading_to(toward_car))
    key_constant(story.hand_target, "location", [(fr(75), stand + Vector((0, 0, 1.2)) - toward_car * 0.35),
                                                 (fr(88), stand + Vector((0, 0, 1.05)) + to_grave * 0.45),
                                                 (fr(92), stand + Vector((0, 0, 1.05)) + toward_car * 0.45)])

    # ---- Flashback: in the road, then digging ----
    impact = layout.car_path.pos(layout.impact_s)
    forward = layout.car_path.tangent(layout.impact_s)
    side = layout.car_path.right(layout.impact_s)
    look_clip = "m_idle_look_around_01"
    looking = prasath.bake(look_clip, first=1, count=min(prasath.clips.length(look_clip), 120), key="look_around")
    prasath.hold(looking, fr(82.2), fr(85) - 1)
    prasath.at(fr(82.2), impact + side * 1.1 + forward * 0.2, heading_to(forward))
    dig_a = prasath.bake("m_idle_nervous_01", first=1, count=2, key="dig_a",
                        tweaks={"Bip01 Spine": Euler((0, 0, 0.42)), "Bip01 Spine1": Euler((0, 0, 0.26)), "Bip01 Head": Euler((0, 0, 0.25))})
    dig_b = prasath.bake("m_idle_nervous_01", first=1, count=2, key="dig_b",
                        tweaks={"Bip01 Spine": Euler((0, 0, 0.12)), "Bip01 Spine1": Euler((0, 0, 0.08)), "Bip01 Head": Euler((0, 0, 0.1))})
    grave = story.grave_position
    dig_stand = grave + toward_car * 1.05
    dig_stand.z = layout.height(dig_stand.x, dig_stand.y)
    prasath.at(fr(85), dig_stand, heading_to(-toward_car))
    prasath.hold(dig_a, fr(85), fr(88) - 1)
    for start in np.arange(85.0, 87.5, 1.2):
        prasath.overlay(dig_b, fr(float(start) + 0.3), fr(float(start) + 0.8), blend=6)
    spade = story.spade
    log("spade size", tuple(round(v, 2) for v in spade.dimensions))
    key_visible(spade, [(fr(85), fr(88) - 1)])
    key_constant(story.hand_target, "location", [(fr(85), dig_stand + Vector((0, 0, 0.72)) - toward_car * 0.5)])
    key_constant(story.hand_target_left, "location", [(fr(85), dig_stand + Vector((0, 0, 0.98)) - toward_car * 0.42)])
    spade.location = dig_stand - toward_car * 0.62 + Vector((0, 0, 0.05))

    # IK arms are only active when his hands have somewhere to be.
    key_constant(ik_right, "influence", [(fr(0), 1.0), (fr(82.2), 0.0), (fr(85), 1.0), (photo + 10, 1.0)])
    key_constant(ik_left, "influence", [(fr(0), 1.0), (fr(65), 0.0), (fr(FLASHBACK[0]), 1.0), (fr(82.2), 0.0), (fr(85), 1.0), (fr(88), 0.0), (photo + 10, 1.0)])

    # ---- Anjali ----
    ghost_stand = anjali.bake("f_idle_neutral_01", first=1, count=2, key="ghost_stand",
                              tweaks={"Bip01 Head": Euler((0.0, 0.34, 0.14)), "Bip01 Neck": Euler((0, 0.12, 0.08))})
    scream = anjali.bake("f_idle_neutral_01", first=1, count=2, key="ghost_scream",
                         tweaks={"Bip01 Head": Euler((0.0, 0.26, -0.15)), "Bip01 Neck": Euler((0, 0.1, -0.05)), "Bip01 MJaw": Euler((0, 0, 0.62)),
                                 "Bip01 LEyeBlinkTop": Euler((0, 0, -0.14)), "Bip01 REyeBlinkTop": Euler((0, 0, -0.14))})
    upright = anjali.bake("f_idle_neutral_01", first=1, count=2, key="normal")
    crying = anjali.bake("f_idle_neutral_01", first=1, count=2, key="crying",
                         tweaks={"Bip01 Head": Euler((0, 0.05, 0.3)), "Bip01 LEyeBlinkTop": Euler((0, 0, 0.3)), "Bip01 REyeBlinkTop": Euler((0, 0, 0.3)),
                                 "Bip01 MJaw": Euler((0, 0, 0.1))})
    riding = anjali.bake(sit, first=1, count=2, key="riding")
    scare_s = float(layout.car_s[fr(34.3)]) + 7.5
    anjali.hold(ghost_stand, fr(32.2), fr(34.7))
    anjali.at(fr(32.2), layout.car_path.pos(scare_s), heading_to(-layout.car_path.tangent(scare_s)))

    # Riding toward the car, then in the road after the impact (face hidden, headlights only).
    bike = story.bicycle
    meet_pos = layout.car_path.pos(layout.impact_s)
    meet_dir = layout.car_path.tangent(layout.impact_s)
    anjali.hold(riding, fr(FLASHBACK[0]), fr(FLASHBACK_IMPACT))
    for frame in range(fr(FLASHBACK[0]), fr(FLASHBACK_IMPACT) + 1, 3):
        seconds = (frame - 1) / FPS
        place = meet_pos + meet_dir * ((FLASHBACK_IMPACT - seconds) * 3.0)
        anjali.root.location = place + Vector((0, 0, 0.22))
        anjali.root.keyframe_insert("location", frame=frame)
        anjali.root.rotation_euler = (0.0, 0.0, heading_to(-meet_dir))
        anjali.root.keyframe_insert("rotation_euler", frame=frame)
        bike.location = place
        bike.keyframe_insert("location", frame=frame)
        bike.rotation_euler = (0.0, 0.0, math.atan2(meet_dir.x, -meet_dir.y))
        bike.keyframe_insert("rotation_euler", frame=frame)
    after = fr(FLASHBACK_IMPACT) + 1
    anjali.hold(upright, after, fr(85) - 1)
    key_constant(anjali.root, "location", [(after, meet_pos + meet_dir * 3.8 - side * 0.3 + Vector((0, 0, 0.13)))])
    key_constant(anjali.root, "rotation_euler", [(after, Euler((math.pi / 2, 0, 1.0)))])
    key_constant(bike, "location", [(after, meet_pos + meet_dir * 5.4 + side * 1.0 + Vector((0, 0, 0.03)))])
    key_constant(bike, "rotation_euler", [(after, Euler((0, math.radians(84), 0.5)))])

    # The flicker approach: each time the light returns she is closer.
    eye = stand + toward_car * 0.15 + Vector((0, 0, 1.04))
    story.pov_eye = eye
    for start, end, distance in ((94.0, 95.2, 11.0), (95.6, 96.6, 7.0), (97.0, 97.9, 3.4), (98.4, 100.0, 0.72)):
        place = Vector((eye.x, eye.y, 0)) + toward_car * distance
        place.z = layout.height(place.x, place.y)
        key_constant(anjali.root, "rotation_euler", [(fr(start), Euler((0.0, 0.0, heading_to(-toward_car))))])
        key_constant(anjali.root, "location", [(fr(start), place)])
        if end == 100.0:
            anjali.hold(ghost_stand, fr(start), fr(99.0) - 1)
            anjali.hold(scream, fr(99.0), fr(100))
        else:
            anjali.hold(ghost_stand, fr(start), fr(end))

    # ---- Police, next morning ----
    kneel_at = grave + toward_car * 0.75
    kneel_at.z = layout.height(kneel_at.x, kneel_at.y) + 0.06
    story.officer_kneel = kneel_at.copy()
    officer.hold(officer.bake("m_crouch_idle", first=1, count=200, key="crouch"), fr(101), fr(115))
    officer.at(fr(101), kneel_at, heading_to(-toward_car))
    second.hold(second.bake(look_clip, first=1, count=min(second.clips.length(look_clip), 120), key="look_around"), fr(101), fr(115))
    second.at(fr(101), parked + tangent * 7.0 + right * -2.0, heading_to(-tangent))

    # ---- Poster photographs (poses held past the end of the film) ----
    anjali.hold(upright, photo, photo + 3, track_name="photo_school")
    anjali.hold(crying, photo + 4, photo + 7, track_name="photo_crying")
    key_constant(anjali.root, "location", [(photo, BOOTH.copy())])
    key_constant(anjali.root, "rotation_euler", [(photo, Euler((0.0, 0.0, 0.0)))])
    photo_a = anjali.bake("f_idle_neutral_01", first=1, count=2, key="behind_a",
                          tweaks={"Bip01 Head": Euler((-1.0, 0.1, 0.2)), "Bip01 Neck": Euler((-0.3, 0, 0))})
    photo_b = anjali.bake("f_idle_neutral_01", first=1, count=2, key="behind_b",
                          tweaks={"Bip01 Head": Euler((0.0, 0.2, 0.05))})
    anjali.hold(photo_a, photo + 10, photo + 13, track_name="photo_a")
    anjali.hold(photo_b, photo + 14, photo + 17, track_name="photo_b")
    prasath.hold(terrified, photo + 10, photo + 17, track_name="photo_v")
    prasath.at(photo + 10, stand, heading_to(toward_car))
    key_constant(anjali.root, "location", [(photo + 10, stand - toward_car * 0.3 + Vector((-toward_car.y, toward_car.x, 0.0)) * 0.36 + Vector((0.0, 0.0, 0.2)))])
    key_constant(anjali.root, "rotation_euler", [(photo + 10, Euler((0.0, 0.0, heading_to(toward_car))))])
    key_constant(story.hand_target, "location", [(photo + 10, stand + Vector((0, 0, 0.95)) + toward_car * 0.1)])
    key_constant(story.hand_target_left, "location", [(photo + 10, stand + Vector((0, 0, 1.0)) + toward_car * 0.1)])

    # ---- Who is visible when ----
    key_visible(prasath.mesh, [(fr(0), fr(94)), (photo + 10, photo + 20)])
    key_visible(anjali.mesh, [(fr(32.2), fr(34.7)), (fr(FLASHBACK[0]), fr(85) - 1), (fr(94), fr(100)), (photo, photo + 20)])
    key_visible(officer.mesh, [(fr(101), fr(115))])
    key_visible(second.mesh, [(fr(101), fr(115))])
    for part in story.phone_parts:
        key_visible(part, [(fr(59.3), fr(94)), (fr(FLASHBACK[0]), fr(82))])
    ghost_keys = [(1, 1.0), (fr(FLASHBACK[0]), 0.0), (fr(85), 1.0), (photo, 0.0), (photo + 10, 1.0)]
    for node in story.ghost_values:
        key_constant(node.outputs[0], "default_value", ghost_keys)


def walker_position(story, seconds):
    """Where Prasath is on his walk to the tree. The cut at 70 s jumps him closer to it."""
    layout = story.layout
    covered = story.walk_speed * (seconds - 65.0)
    if seconds >= 70.0:
        covered += max(story.walk_distance - story.walk_speed * 10.0, 0.0)
    place = story.walk_start + story.walk_direction * min(covered, story.walk_distance)
    place.z = layout.height(place.x, place.y)
    return place


def direct_props_and_light(story):
    layout, car = story.layout, story.car
    fb0, fb1, impact = fr(FLASHBACK[0]), fr(FLASHBACK[1]), fr(FLASHBACK_IMPACT)
    photo = PHOTO_FRAME
    toward_car = story.toward_car

    # Posters do not exist yet in the flashback.
    for poster in story.posters:
        key_visible(poster, [(1, fb0 - 1), (fb1, LAST_FRAME + 2), (photo, photo + 40)])
    key_visible(story.poster_digging, [(fr(60), fr(80) - 1)])
    key_visible(story.poster_prasath, [(fr(101), LAST_FRAME + 2)])
    key_constant(story.poster_snap.outputs[0], "default_value", [(1, 0.0), (fr(118.0), 1.0)])

    # He tears the last poster off the tree.
    base = story.poster_digging.location.copy()
    key_values(story.poster_digging, "location", [(fr(77.4), base), (fr(77.8), base + toward_car * 0.34 + Vector((0, 0, -0.1)))], kind="LINEAR")

    # Grave and shoe
    key_values(story.grave_sunk, "value", [(1, 0.0), (fr(88.3), 0.0), (fr(91.5), 1.0)], kind="LINEAR")
    shoe = story.shoe
    resting = shoe.location.copy()
    key_values(shoe, "location", [(fr(88.3), resting - Vector((0, 0, 0.22))), (fr(90.4), resting)], kind="LINEAR")
    key_visible(shoe, [(fr(88.3), LAST_FRAME + 2)])

    # Door: closed while he drives, open once he leaves it, closed again in the flashback.
    open_windows = [(fr(65), fb0 - 1), (fb1, LAST_FRAME + 2)]
    key_visible(car.shell_open, open_windows)
    key_visible(car.door, open_windows)
    key_visible(car.window_open, open_windows)
    closed = [(1, fr(65) - 1), (fb0, fb1 - 1), (photo, photo + 40)]
    key_visible(car.shell, closed)
    key_visible(car.objects["Window"], closed)

    # Wipers run while the engine runs. Next morning they are still going, with nobody in the car.
    def morning_clock(frame):
        return 88.0 + (frame - fr(101)) / FPS

    clock_keys = [(fr(0), 0.0), (fr(55.5), 55.5), (fb0 - 1, 55.5), (fb0, 80.0), (fb1, 88.0), (fr(101), 88.0), (LAST_FRAME + 2, morning_clock(LAST_FRAME + 2))]
    key_values(car.wipe_clock.outputs[0], "default_value", clock_keys, kind="LINEAR")

    def stamp_wipers(frames, clock_of):
        for frame in frames:
            tau = (clock_of(frame) / WIPER_PERIOD) % 1.0
            angle = WIPER_SWEEP * (1.0 - math.cos(math.tau * tau)) / 2.0
            for sweep in car.wiper_sweeps:
                key_values(sweep, "rotation_euler", [(frame, angle)], kind="LINEAR" if frame != frames[-1] else "CONSTANT", index=2)

    stamp_wipers(range(fr(0), fr(55.5), 3), lambda f: (f - 1) / FPS)
    stamp_wipers(range(fb0, fb1, 3), lambda f: (f - 1) / FPS)
    stamp_wipers(range(fr(101), LAST_FRAME + 2, 3), morning_clock)

    # Steering wheel follows the road, gently.
    for frame in range(fr(0), fr(59), 6):
        here = layout.car_frame(frame)[1]
        ahead = layout.car_frame(frame + 12)[1]
        turn = math.atan2(here.x * ahead.y - here.y * ahead.x, here.dot(ahead))
        car.wheel_spin.rotation_euler = (0.0, 0.0, -turn * 4.0)
        car.wheel_spin.keyframe_insert("rotation_euler", frame=frame, index=2)

    # Phone screen faces him whenever he looks at it.
    def face_driver(frame):
        eye = car_local_to_world(layout, frame, (SEAT_X, -0.05, 1.10))
        phone = car_local_to_world(layout, frame, (SEAT_X - 0.05, 0.16, 0.98))
        key_constant(story.phone, "rotation_quaternion", [(frame, (eye - phone).normalized().to_track_quat("Z", "Y"))])

    face_driver(fr(59.3))
    face_driver(fb0)

    # Car lights
    good, cracked = 850.0, 520.0
    pattern = (1.0, 0.25, 0.9, 0.05, 0.7, 0.0, 0.55, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0)
    flicker = [(fr(55.2 + index * 0.3), level) for index, level in enumerate(pattern)]

    def powered(base, off=0.0, morning=0.0):
        return [(1, base)] + [(frame, base * level) for frame, level in flicker] + [(fr(59.2), off), (fb0, base), (fb1, 0.0), (fr(101), morning)]

    key_constant(car.head_right.data, "energy", powered(good, morning=good * 0.2))
    key_constant(car.head_left.data, "energy", powered(cracked, morning=cracked * 0.2))
    tail_keys = [(1, 2.2), (fr(13), 6.0), (fr(19), 2.2), (fr(33.0), 8.0), (fr(38), 2.2), (fr(59.2), 0.0), (fb0, 2.2), (impact, 8.0), (fb1, 0.0), (fr(101), 0.6)]
    for lamp in car.tail_lamps:
        key_constant(lamp.data, "energy", tail_keys)
    key_constant(car.cluster_glow, "default_value", powered(1.4, off=0.15))
    key_constant(car.radio_glow, "default_value", powered(1.2, off=0.15))
    key_constant(car.dash_light.data, "energy", powered(14.0))

    # Soft light on the roadside so the posters can be read through the side window.
    fill = bpy.data.lights.new("PosterFill", "SPOT")
    fill.spot_size = math.radians(70)
    fill.spot_blend = 1.0
    fill.shadow_soft_size = 0.4
    fill.color = (1.0, 0.86, 0.68)
    fill_object = unseen(add_object("PosterFill", fill, get_collection("Car")))
    fill_object.parent = car.root
    fill_object.location = (0.7, 0.8, 1.55)
    fill_object.rotation_mode = "QUATERNION"
    fill_object.rotation_quaternion = look_at((0.7, 0.8, 1.55), (5.0, 1.6, 1.3))
    key_constant(fill, "energy", [(1, 0.0), (fr(38.0), 42.0), (fr(51.0), 0.0), (fr(59.2), 60.0), (fr(65), 0.0)])

    # Cold night light in the cabin, only during shots inside the car.
    cabin = bpy.data.lights.new("CabinFill", "AREA")
    cabin.size = 0.6
    cabin.color = (0.62, 0.74, 1.0)
    cabin_object = unseen(add_object("CabinFill", cabin, get_collection("Car")))
    cabin_object.parent = car.root
    cabin_object.location = (0.1, -0.1, 1.27)
    cabin_keys = [(1, 0.0)]
    for start, end in ((8.0, 12.0), (16.0, 19.0), (26.8, 29.0), (51.0, 55.0), (59.0, 65.0), (FLASHBACK[0], 82.2)):
        cabin_keys += [(fr(start), 9.0), (fr(end), 0.0)]
    key_constant(cabin, "energy", cabin_keys)

    # The phone screen lights his hand and face.
    screen_glow = bpy.data.lights.new("PhoneGlow", "POINT")
    screen_glow.color = (0.72, 0.84, 1.0)
    screen_glow.shadow_soft_size = 0.05
    screen_object = unseen(add_object("PhoneGlow", screen_glow, get_collection("Props")))
    screen_object.parent = story.hand_target
    screen_object.location = (0.0, 0.0, 0.04)
    key_constant(screen_glow, "energy", [(1, 0.0), (fr(59.3), 1.6), (fr(65), 0.0), (fb0, 1.6), (fr(82.0), 0.0)])

    # Cold breath in the back seat
    key_values(car.breath_amount.outputs[0], "default_value", [(1, 0.0), (fr(51.5), 0.0), (fr(54.0), 16.0), (fr(55.0), 0.0)], kind="LINEAR")
    key_values(car.back_light.data, "energy", [(1, 0.0), (fr(51.0), 0.0), (fr(53.0), 8.0), (fr(55.0), 0.0)], kind="LINEAR")

    # Handprint: appears in one frame with the slam.
    key_visible(story.handprint, [(fr(63.2), fr(65))])

    # Phone torch: on from the moment he leaves the car, failing in the final scare.
    torch = story.torch
    key_constant(torch.data, "energy", [(1, 0.0), (fr(65), 90.0), (fr(75), 22.0), (fb0, 0.0), (fr(85), 40.0), (fr(88), 70.0), (fr(92), 70.0), (fr(94), 90.0), (fr(95.2), 0.0), (fr(95.6), 80.0),
                                        (fr(96.6), 0.0), (fr(97.0), 40.0), (fr(97.9), 0.0), (fr(98.4), 14.0), (fr(100), 0.0)])

    def aim(frame, origin, target):
        key_constant(torch, "rotation_quaternion", [(frame, look_at(origin, target))])

    stand, grave = story.stand, story.grave_position
    hand_height = Vector((0, 0, 1.1))
    aim(fr(65), layout.parked_pos + hand_height, stand + Vector((0, 0, 0.6)))
    aim(fr(75), stand + hand_height, story.bark_point)
    aim(fr(85), grave + toward_car * 0.9 + Vector((0, 0, 0.7)), grave)
    aim(fr(88), stand + hand_height, grave)
    aim(fr(92), stand + hand_height, stand + toward_car * 8.0 + Vector((0, 0, 0.9)))
    aim(fr(94), story.pov_eye, story.pov_eye + toward_car * 8.0 + Vector((0, 0, -0.2)))

    # Weather
    morning = fr(101)
    # Photo frames past the end of the film go back to night, because those pictures are taken at night.
    key_constant(story.world_mix, "default_value", [(1, 0.0), (morning, 1.0), (photo, 0.0)])
    key_constant(story.night_strength, "default_value", [(1, 0.55)])
    key_constant(story.fog_density, "default_value", [(1, 0.0045), (morning, 0.038), (photo, 0.0045)])
    key_constant(story.fog_glow, "default_value", [(1, 0.0), (morning, 0.014), (photo, 0.0)])
    key_constant(story.fog_object, "scale", [(1, Vector((1, 1, 1))), (morning, Vector((1, 1, 0.16))), (photo, Vector((1, 1, 1)))])
    key_visible(story.rain, [(1, fr(100) - 1), (photo + 10, photo + 40)])

    # Exposure carries the white impact flash and the shift to morning.
    key_constant(bpy.context.scene.view_settings, "exposure", [(1, 0.35), (impact, 4.5), (impact + 2, 2.0), (impact + 3, 0.35), (morning, -0.15), (photo, 0.35)])

    # Red and blue police lights glowing through the morning fog.
    parked, tangent = layout.parked_pos, layout.car_path.tangent(layout.parked_s)
    for name, color, phase, shift in (("PoliceRed", (1.0, 0.03, 0.02), 0, -0.7), ("PoliceBlue", (0.03, 0.12, 1.0), 1, 0.7)):
        lamp = bpy.data.lights.new(name, "POINT")
        lamp.color = color
        lamp.shadow_soft_size = 0.3
        obj = unseen(add_object(name, lamp, get_collection("Set")))
        obj.location = parked + tangent * 19.0 + Vector((shift, 0.0, 1.6))
        keys = [(1, 0.0)]
        frame = morning
        on = phase == 0
        while frame < fr(115):
            keys.append((frame, 500.0 if on else 0.0))
            frame += 7
            on = not on
        keys.append((fr(115), 0.0))
        key_constant(lamp, "energy", keys)


# ---------------------------------------------------------------------------
# Camera work: every shot in the film
# ---------------------------------------------------------------------------
BLACK_CARD = Vector((3000.0, 0.0, 0.0))
TITLE_CARD = Vector((3000.0, 50.0, 0.0))


class ShotList:
    def __init__(self):
        self.scene = bpy.context.scene
        self.coll = get_collection("Cameras")
        self.shots = []

    def add(self, name, t0, t1, func, lens=35.0, fstop=2.8, parent=None, shake=0.0, step=2):
        data = bpy.data.cameras.new(name)
        data.lens = lens
        data.sensor_width = 36.0
        data.clip_start = 0.02
        data.clip_end = 1600.0
        data.dof.use_dof = True
        data.dof.aperture_fstop = fstop
        camera = add_object(f"CAM_{name}", data, self.coll)
        camera.parent = parent
        camera.rotation_mode = "QUATERNION"
        marker = self.scene.timeline_markers.new(name, frame=fr(t0))
        marker.camera = camera
        seed = len(self.shots) * 3.7
        for frame in range(fr(t0) - 1, fr(t1) + 2, step):
            seconds = (frame - 1) / FPS
            location, target = func(seconds)
            location, target = Vector(location), Vector(target)
            focus = (target - location).length
            if shake:
                amount = shake * 0.012
                location = location + Vector((mnoise.noise(Vector((seconds * 1.1, seed, 0.0))), mnoise.noise(Vector((seconds * 0.9, seed, 3.0))),
                                              mnoise.noise(Vector((seconds * 1.3, seed, 7.0))))) * amount
                target = target + Vector((mnoise.noise(Vector((seconds * 0.7, seed, 11.0))), mnoise.noise(Vector((seconds * 0.8, seed, 13.0))),
                                          mnoise.noise(Vector((seconds * 0.6, seed, 17.0))))) * amount * max(focus, 1.0) * 0.8
            camera.location = location
            camera.keyframe_insert("location", frame=frame)
            camera.rotation_quaternion = look_at(location, target)
            camera.keyframe_insert("rotation_quaternion", frame=frame)
            data.dof.focus_distance = focus
            data.dof.keyframe_insert("focus_distance", frame=frame)
        self.shots.append((name, t0, t1, camera))
        return camera


def build_shots(story):
    layout, car = story.layout, story.car
    shots = ShotList()
    eye = (SEAT_X, -0.05, 1.10)
    toward_car = story.toward_car
    across = Vector((-toward_car.y, toward_car.x, 0.0))

    def at_car(seconds, x, y, z):
        return car_local_to_world(layout, fr(seconds), (x, y, z))

    def ease(a, b, t):
        return float(smoothstep(a, b, t))

    # --- The highway ---
    shots.add("01_aerial", 0, 5, lambda t: (at_car(t, 7.0 + t, -30.0 + t * 1.5, 24.0 - t), at_car(t, 0, 7, 0.5)), lens=30, fstop=8, shake=1.2)
    shots.add("02_headlight", 5, 8, lambda t: (at_car(t, -1.75, 3.7, 0.62), at_car(t, -0.56, 2.05, 0.68)), lens=40, fstop=2.2, shake=0.7)
    shots.add("03_driver", 8, 10.5, lambda t: ((-0.34, 0.42 + (t - 8) * 0.02, 1.02), (SEAT_X, -0.05, 1.12)), lens=28, fstop=2.0, parent=car.root, shake=0.5)
    shots.add("04_date", 10.5, 12, lambda t: ((SEAT_X + 0.03, 0.40 + (t - 10.5) * 0.06, 0.95), (SEAT_X, 0.755, 0.885)), lens=42, fstop=2.2, parent=car.root, shake=0.3)
    shots.add("05_closure", 12, 16, lambda t: (eye, (SEAT_X - 0.6, 30.0, 0.4)), lens=24, fstop=4, parent=car.root, shake=0.5)
    shots.add("06_stare", 16, 19, lambda t: ((SEAT_X - 0.06, 0.40, 1.02), (SEAT_X, -0.05, 1.13)), lens=42, fstop=1.8, parent=car.root, shake=0.4)
    sign_view = Vector((10.5, 77.0, 1.3))
    shots.add("07_turnoff", 19, 22, lambda t: (sign_view, at_car(t, 0, 1, 0.7)), lens=32, fstop=5.6, shake=0.6)

    # --- Old Forest Road ---
    old_poster = story.hero_posters["old"]
    old = old_poster.location.copy()
    old_view = old + old_poster.rotation_quaternion @ Vector((0.12, -0.05, 1.05))
    shots.add("08_forest_pov", 22, 25, lambda t: (at_car(t, *eye), at_car(t, 0.2 + 0.8 * ease(23.6, 24.8, t), 20.0, 0.6)), lens=24, fstop=4, shake=0.8)
    shots.add("09_old_poster", 25, 26.8, lambda t: (old_view, old), lens=45, fstop=2.8, shake=0.5)
    shots.add("10_looks_away", 26.8, 29, lambda t: ((-0.34, 0.42, 1.02), (SEAT_X, -0.05, 1.12)), lens=30, fstop=2.0, parent=car.root, shake=0.5)
    shots.add("11_every_tree", 29, 32, lambda t: (at_car(t, *eye), at_car(t, 0.2, 22.0, 0.8)), lens=22, fstop=4, shake=0.9)

    # --- Jump scare 1 ---
    shots.add("12_girl_in_road", 32, 33.8, lambda t: (at_car(t, *eye), at_car(t, 0.0, 20.0, 0.7)), lens=26, fstop=4, shake=2.4)
    tyre_view = at_car(33.2, -2.4, -1.4, 0.26)
    shots.add("13_brakes", 33.8, 35, lambda t: (tyre_view, at_car(t, -0.7, -1.2, 0.34)), lens=24, fstop=2.8, shake=1.3)
    shots.add("14_gone", 35, 38, lambda t: (at_car(t, *eye), at_car(t, 0.0, 20.0, 0.7)), lens=26, fstop=4, shake=0.7)

    # --- The posters change ---
    def side_window(key):
        poster = story.hero_posters[key].location.copy()
        return lambda t: (at_car(t, SEAT_X + 0.22, -0.02, 1.10), poster)

    shots.add("15_poster_smile", 38, 42, side_window("smile"), lens=48, fstop=2.8, shake=0.6)
    shots.add("16_poster_crying", 42, 45, side_window("crying"), lens=52, fstop=2.8, shake=0.6)
    shots.add("17_poster_blood", 45, 48, side_window("blood"), lens=56, fstop=2.8, shake=0.6)
    shots.add("18_poster_road", 48, 51, side_window("road"), lens=62, fstop=2.8, shake=0.6)
    mirror = Vector(MIRROR_LOCATION)
    shots.add("19_mirror", 51, 55, lambda t: ((SEAT_X - 0.04, -0.02, 1.09), mirror), lens=68, fstop=2.0, parent=car.root, shake=0.3)

    # --- Engine dies, slam ---
    dead_view = at_car(55, 3.8, 15.0, 0.9)
    shots.add("20_engine_dies", 55, 59, lambda t: (dead_view, at_car(t, 0, 0.5, 0.75)), lens=38, fstop=4, shake=0.5)
    shots.add("21_no_signal", 59, 63, lambda t: ((SEAT_X - 0.34, -0.20, 1.14), (SEAT_X - 0.05, 0.16, 0.98)), lens=38, fstop=2.0, parent=car.root, shake=0.5)
    shots.add("22_slam", 63, 65, lambda t: ((-0.34, 0.30, 1.05), (SEAT_X + 0.2, -0.03, 1.09)), lens=28, fstop=2.2, parent=car.root, shake=0.8)

    # --- Out into the trees ---
    direction = story.walk_direction
    shots.add("23_gets_out", 65, 70, lambda t: (walker_position(story, t) - direction * 2.6 + Vector((0.3, 0.3, 1.55)),
                                                walker_position(story, t) + direction * 1.5 + Vector((0, 0, 1.15))), lens=30, fstop=2.8, shake=1.1)
    wide = story.stand - direction * 3.0 - Vector((-direction.y, direction.x, 0.0)) * 7.5 + Vector((0, 0, 1.35))
    shots.add("24_straight_to_tree", 70, 75, lambda t: (wide, walker_position(story, t) + Vector((0, 0, 1.1))), lens=30, fstop=3.2, shake=0.7)
    poster_spot = story.bark_point + toward_car * 0.02
    shots.add("25_poster_truth", 75, 80, lambda t: (poster_spot + toward_car * (0.92 - (t - 75.0) * 0.05) + Vector((0.1, 0.05, 0.05)), poster_spot + Vector((0, 0, 0.02))), lens=44, fstop=2.8, shake=0.6)

    # --- One year ago ---
    shots.add("26_fb_pov", 80, 82.2, lambda t: (at_car(t, *eye), at_car(t, 0.0, 22.0, 0.5)), lens=26, fstop=4, shake=0.9)
    impact_pos = layout.car_path.pos(layout.impact_s)
    forward = layout.car_path.tangent(layout.impact_s)
    side = layout.car_path.right(layout.impact_s)
    shots.add("27_fb_road", 82.2, 85, lambda t: (impact_pos - forward * 7.5 - side * 2.4 + Vector((0, 0, 0.62)), impact_pos + forward * 3.4 + Vector((0, 0, 0.45))), lens=30, fstop=4, shake=0.8)
    grave = story.grave_position
    dig_view = grave + toward_car * 3.6 + across * 3.4 + Vector((0, 0, 0.8))
    shots.add("28_fb_digging", 85, 88, lambda t: (dig_view, grave + toward_car * 0.9 + Vector((0, 0, 0.8))), lens=38, fstop=3.2, shake=0.8)

    # --- The climax ---
    shoe_view = grave + toward_car * 0.65 - across * 0.3 + Vector((0, 0, 0.32))
    shots.add("29_shoe", 88, 92, lambda t: (shoe_view, grave + Vector((0.05, 0, 0.1))), lens=45, fstop=2.8, shake=0.4)
    stand = story.stand
    shots.add("30_turns", 92, 94, lambda t: (stand + toward_car * 1.05 + Vector((0.15, 0.1, 1.48)), stand + Vector((0, 0, 1.52))), lens=38, fstop=2.0, shake=1.0)
    pov = story.pov_eye

    def falling(t):
        drop = 0.42 * ease(94.0, 94.5, t)
        return pov + Vector((0, 0, -drop)), pov + toward_car * 8.0 + Vector((0, 0, 0.1 + drop * 0.5))

    shots.add("31_flicker", 94, 98.4, falling, lens=30, fstop=2.8, shake=2.2)
    face = pov + toward_car * 0.72 + Vector((0, 0, 0.27))
    shots.add("32_scream", 98.4, 100, lambda t: (pov + Vector((0, 0, 0.2)), face), lens=30, fstop=2.0, shake=2.6)

    # --- Silence ---
    shots.add("33_black", 100, 101, lambda t: (BLACK_CARD + Vector((0, -1, 0)), BLACK_CARD), lens=35, fstop=22)

    # --- Next morning ---
    parked = layout.parked_pos
    front_view = parked + layout.car_path.tangent(layout.parked_s) * 9.5 + layout.car_path.right(layout.parked_s) * 4.2 + Vector((0, 0, 0.85))
    shots.add("34_empty_car", 101, 107, lambda t: (front_view, parked + Vector((0, 0, 0.8))), lens=38, fstop=5.6, shake=0.45)
    kneel = story.officer_kneel
    shots.add("35_we_found_her", 107, 111, lambda t: (grave + toward_car * 1.9 - across * 1.9 + Vector((0, 0, 0.55)), kneel + Vector((0, 0, 0.5))), lens=40, fstop=2.4, shake=0.5)

    def trunk_pan(t):
        progress = ease(111.0, 115.0, t)
        return grave + toward_car * 2.8 + Vector((0.15, 0.1, 0.45 + 0.9 * progress)), grave.lerp(story.bark_point, progress)

    shots.add("36_pan_up", 111, 115, trunk_pan, lens=38, fstop=4, shake=0.4)
    photo_centre = story.poster_prasath.location + Vector((0, 0, 0.03))

    def push_in(t):
        return photo_centre + toward_car * (1.0 - 0.62 * ease(115.0, 118.5, t)), photo_centre

    shots.add("37_eyes_snap", 115, 118.5, push_in, lens=42, fstop=4, shake=0.25)
    shots.add("38_title", 118.5, 120.05, lambda t: (TITLE_CARD + Vector((0, -1, 0)), TITLE_CARD), lens=36, fstop=22)
    return shots


def build_end_cards(coll):
    black = bpy.data.materials.new("PureBlack")
    nb = Nodes(black.node_tree)
    out = nb.new("ShaderNodeOutputMaterial")
    nb.wire(nb.new("ShaderNodeEmission", Strength=0.0).outputs[0], out.inputs["Surface"])
    image_plane("BlackCard", 6.0, 4.0, black, coll, None, BLACK_CARD, (math.radians(90), 0, 0))
    canvas = Canvas(1920, 1080, (0.0, 0.0, 0.0))
    canvas.text("LAST SEEN", 960, 495, 165, (0.86, 0.84, 0.80), font="impact", align="center")
    material, _ = emission_image_material("TitleImage", canvas.save("title_card"), 1.0)
    image_plane("TitleCard", 1.0, 0.5625, material, coll, None, TITLE_CARD, (math.radians(90), 0, 0))
    image_plane("TitleSurround", 6.0, 4.0, black, coll, None, TITLE_CARD + Vector((0, 0.05, 0)), (math.radians(90), 0, 0))


# ---------------------------------------------------------------------------
# Film look: lens, glow, grade
# ---------------------------------------------------------------------------
def build_compositor(scene):
    tree = bpy.data.node_groups.new("FilmLook", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    nodes, links = tree.nodes, tree.links
    source = nodes.new("CompositorNodeRLayers")
    lens = nodes.new("CompositorNodeLensdist")
    lens.inputs["Distortion"].default_value = -0.012
    lens.inputs["Dispersion"].default_value = 0.012
    links.new(source.outputs["Image"], lens.inputs["Image"])
    glare = nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Bloom"
    glare.inputs["Threshold"].default_value = 1.0
    glare.inputs["Strength"].default_value = 0.35
    glare.inputs["Size"].default_value = 0.7
    links.new(lens.outputs["Image"], glare.inputs["Image"])
    grade = nodes.new("CompositorNodeHueSat")
    links.new(glare.outputs["Image"], grade.inputs["Image"])
    output = nodes.new("NodeGroupOutput")
    links.new(grade.outputs["Image"], output.inputs[0])
    key_constant(grade.inputs["Saturation"], "default_value", [(1, 0.88), (fr(FLASHBACK[0]), 0.22), (fr(FLASHBACK[1]), 0.88), (fr(101), 0.7)])
    scene.render.use_compositing = True


# ---------------------------------------------------------------------------
# Poster photographs, rendered from the film's own characters and locations
# ---------------------------------------------------------------------------
def eye_positions(actor, scene, camera, frame):
    """Eye positions in the photo (0..1), used to paint tears in the right place."""
    saved = (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage)
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 640, 670, 100
    scene.frame_set(frame)
    depsgraph = scene.view_layers[0].depsgraph
    depsgraph.update()
    evaluated = actor.arm.evaluated_get(depsgraph)
    points = []
    for name in ("Bip01 LEye", "Bip01 REye"):
        co = world_to_camera_view(scene, camera, evaluated.matrix_world @ evaluated.pose.bones[name].head)
        points.append((co.x, co.y))
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = saved
    return points


def render_still(scene, camera, path, frame, samples, width=640, height=670):
    render = scene.render
    saved = (render.resolution_x, render.resolution_y, render.resolution_percentage, scene.cycles.samples, render.filepath,
             render.use_compositing, render.use_motion_blur, render.use_persistent_data)
    render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
    scene.cycles.samples = samples
    render.filepath = str(path)
    render.use_compositing = False
    render.use_motion_blur = False
    render.use_persistent_data = False
    scene.frame_set(frame)
    scene.camera = camera
    bpy.ops.render.render(write_still=True, scene=scene.name)
    (render.resolution_x, render.resolution_y, render.resolution_percentage, scene.cycles.samples, render.filepath,
     render.use_compositing, render.use_motion_blur, render.use_persistent_data) = saved


def take_poster_photos(story, force=False):
    wanted = ("school", "crying", "blood", "road", "digging", "prasath_a", "prasath_b")
    if not force and all((GENERATED / f"photo_{name}.png").exists() for name in wanted):
        log("poster photographs already exist")
        return False
    main_scene = bpy.context.scene
    layout = story.layout
    anjali = story.anjali
    photo = PHOTO_FRAME

    # Markers must not switch the camera while these photographs are taken.
    bound = [(m, m.camera) for m in main_scene.timeline_markers]
    for marker, _ in bound:
        marker.camera = None

    # --- School portrait, in a small photo booth outside the forest ---
    booth = get_collection("Booth")
    backdrop_mesh = bpy.data.meshes.new("BoothBackdrop")
    backdrop_mesh.from_pydata([(-3, 0.9, -1), (3, 0.9, -1), (3, 0.9, 3), (-3, 0.9, 3)], [], [(0, 1, 2, 3)])
    backdrop_mesh.materials.append(leather_material("BoothBackdrop", (0.14, 0.24, 0.42), 0.8))
    backdrop = add_object("BoothBackdrop", backdrop_mesh, booth)
    backdrop.location = BOOTH
    temporary = [backdrop]

    def booth_light(name, size, energy, offset):
        light = bpy.data.lights.new(name, "AREA")
        light.size = size
        light.energy = energy
        obj = unseen(add_object(name, light, booth))
        obj.location = BOOTH + Vector(offset)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = look_at(obj.location, BOOTH + Vector((0, 0, 1.2)))
        temporary.append(obj)

    booth_light("BoothKey", 1.6, 34.0, (0.9, -1.8, 1.9))
    booth_light("BoothFill", 1.4, 12.0, (-1.2, -1.6, 1.3))
    camera_data = bpy.data.cameras.new("BoothCam")
    camera_data.lens = 85.0
    camera = add_object("BoothCam", camera_data, booth)
    camera.location = BOOTH + Vector((0.0, -1.55, 1.27))
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = look_at(camera.location, BOOTH + Vector((0.0, 0.0, 1.20)))
    temporary.append(camera)
    main_scene.camera = camera

    crying_path = GENERATED / "photo_crying.png"
    render_still(main_scene, camera, GENERATED / "photo_school.png", photo + 1, 48)
    render_still(main_scene, camera, crying_path, photo + 5, 48)

    # Tears, red eyes and the darker third photograph are painted over the crying portrait.
    rng = np.random.default_rng(21)
    picture = read_image(crying_path)
    height_px, width_px = picture.shape[:2]
    yy, xx = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
    eyes = eye_positions(anjali, main_scene, camera, photo + 5)
    tears = np.zeros((height_px, width_px), dtype=np.float32)
    redness = np.zeros((height_px, width_px), dtype=np.float32)
    for ex, ey in eyes:
        cx, cy = ex * width_px, ey * height_px
        redness = np.maximum(redness, np.exp(-(((xx - cx) / 34.0) ** 2 + ((yy - cy) / 20.0) ** 2)))
        for offset in (-7.0, 4.0):
            track = cx + offset + 3.0 * np.sin(yy * 0.05 + offset)
            tears = np.maximum(tears, np.exp(-((xx - track) / 2.2) ** 2) * (yy < cy - 5) * (yy > cy - 125))
    crying = picture * (0.8 - 0.1 * blotches(height_px, width_px, 6, rng))[..., None]
    crying = crying * (1 - 0.55 * redness[..., None]) + np.array((0.5, 0.08, 0.07), dtype=np.float32) * 0.55 * redness[..., None]
    crying = crying + tears[..., None] * 0.28
    canvas = Canvas(width_px, height_px)
    canvas.rgb = np.clip(crying, 0, 1).astype(np.float32)
    canvas.save("photo_crying")
    stain_shape = np.clip((blotches(height_px, width_px, 16, rng) - 0.42) * 5.0, 0, 1)
    fx, fy = sum(p[0] for p in eyes) / 2 * width_px, sum(p[1] for p in eyes) / 2 * height_px
    near_face = np.exp(-(((xx - fx) / 120.0) ** 2 + ((yy - fy) / 165.0) ** 2))
    stain = np.clip(stain_shape * near_face * 1.25, 0, 1)[..., None]
    canvas.rgb = np.clip(crying * 0.5 * (1 - stain * 0.92) + np.array((0.16, 0.012, 0.012), dtype=np.float32) * stain * 0.92, 0, 1).astype(np.float32)
    canvas.save("photo_blood")
    for obj in temporary:
        bpy.data.objects.remove(obj)
    bpy.data.collections.remove(booth)

    # --- Pictures taken inside the film's own locations ---
    coll = get_collection("Cameras")
    data = bpy.data.cameras.new("PhotoCam")
    data.clip_start = 0.02
    cam = add_object("PhotoCam", data, coll)
    cam.rotation_mode = "QUATERNION"
    flash = bpy.data.lights.new("PhotoFlash", "SPOT")
    flash.spot_size = math.radians(80)
    flash.spot_blend = 0.8
    flash.shadow_soft_size = 0.05
    flash_object = unseen(add_object("PhotoFlash", flash, coll))
    flash_object.rotation_mode = "QUATERNION"
    main_scene.camera = cam

    def shoot(name, frame, location, target, lens, energy):
        data.lens = lens
        cam.location = location
        cam.rotation_quaternion = look_at(location, target)
        flash_object.location = location
        flash_object.rotation_quaternion = cam.rotation_quaternion
        flash.energy = energy
        render_still(main_scene, cam, GENERATED / f"photo_{name}.png", frame, 40)

    toward_car = story.toward_car
    across = Vector((-toward_car.y, toward_car.x, 0))
    road_frame = fr(46.0)
    shoot("road", road_frame, car_local_to_world(layout, road_frame, (0.0, -0.58, 1.08)),
          car_local_to_world(layout, road_frame, (0.05, 14.0, 0.55)), 24.0, 0.0)
    grave = story.grave_position
    shoot("digging", fr(86.2), grave + toward_car * 3.7 + across * 1.2 + Vector((0, 0, 1.25)),
          grave + toward_car * 0.9 + Vector((0, 0, 0.75)), 40.0, 130.0)
    face = story.stand + Vector((0.05, 0, 1.4))
    spot = story.stand + toward_car * 1.25 + Vector((0.08, 0.06, 1.48))
    shoot("prasath_a", photo + 12, spot, face, 42.0, 40.0)
    shoot("prasath_b", photo + 16, spot, face, 42.0, 40.0)
    # Prints fade: lift the dark night photographs so they read on paper in the rain.
    for name, gain in (("road", 2.6), ("digging", 1.5), ("prasath_a", 1.25), ("prasath_b", 1.25)):
        path = GENERATED / f"photo_{name}.png"
        picture = read_image(path)
        canvas = Canvas(picture.shape[1], picture.shape[0])
        canvas.rgb = np.clip(picture * gain, 0, 1).astype(np.float32)
        canvas.save(f"photo_{name}")
    for marker, camera_object in bound:
        marker.camera = camera_object
    bpy.data.objects.remove(cam)
    bpy.data.objects.remove(flash_object)
    bpy.data.cameras.remove(data)
    bpy.data.lights.remove(flash)
    log("poster photographs rendered")
    return True


# ---------------------------------------------------------------------------
# Build everything
# ---------------------------------------------------------------------------
def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.actions, bpy.data.armatures, bpy.data.cameras, bpy.data.lights,
                  bpy.data.curves, bpy.data.node_groups, bpy.data.worlds):
        for item in list(block):
            block.remove(item)
    bpy.context.scene.timeline_markers.clear()


def write_credits():
    (PROJECT / "CREDITS.txt").write_text(
        "LAST SEEN - asset credits\n\n"
        "BMW 1M model: \"BMW27\" Cycles benchmark scene by Mike Pan, from the Blender Foundation demo files.\n"
        "  https://download.blender.org/demo/test/  Check its licence and BMW's trademark before commercial use.\n"
        "People and motion capture: Microsoft Rocketbox Avatar Library, MIT licence.\n"
        "  https://github.com/microsoft/Microsoft-Rocketbox  (licence text in assets/rocketbox/LICENSE.txt)\n"
        "Trees, plants, rocks, road and ground textures, skies, spade, road barriers: Poly Haven, CC0.\n"
        "  https://polyhaven.com\n", encoding="utf-8")


def build_film(rebuild_photos=False):
    started = time.time()
    clear_scene()
    scene = bpy.context.scene
    scene.name = "LastSeen"
    configure_render(scene)
    layout = Layout()
    story = Story()
    story.layout = layout
    GENERATED.mkdir(parents=True, exist_ok=True)
    story.images = {"sign": paint_detour_sign(), "dash": paint_dashboard(), "radio": paint_radio(), "phone": paint_phone(),
                    "plate": paint_plate(), "handprint": paint_handprint(), "posters": paint_posters()}

    world_coll = get_collection("World")
    build_road("Highway", layout.highway, HIGHWAY_HALF_WIDTH, build_road_material("RoadHighway", True), world_coll, 0.0, False)
    build_road("ForestRoad", layout.forest, FOREST_HALF_WIDTH, build_road_material("RoadForest", False), world_coll, -0.015, True)
    build_terrain(layout, world_coll)
    get_collection("_Sources")
    story.rain = build_rain(world_coll)
    story.models = build_forest(layout)
    story.fog_object, volume = build_fog(world_coll)
    story.fog_density = volume.inputs["Density"]
    story.fog_glow = volume.inputs["Emission Strength"]
    story.world_mix, story.night_strength = build_world()
    log(f"world built in {time.time() - started:.0f}s")

    car_coll = get_collection("Car")
    story.car = build_car(layout, car_coll)
    build_cockpit(story.car, story.images, car_coll)
    build_wipers(story.car, car_coll)
    cut_open_door(story.car, car_coll)
    animate_car(story.car, layout)

    people = get_collection("People")
    clips = ClipLibrary()
    story.prasath = Actor("Prasath", "Male_Adult_07", people, clips)
    story.anjali = Actor("Anjali", "Female_Child_02", people, clips)
    story.officer = Actor("Officer", "Police_Male_01", people, clips)
    story.second_officer = Actor("Constable", "Police_Male_01", people, clips)
    make_khaki_uniform(story.officer)
    make_khaki_uniform(story.second_officer)
    story.ghost_values = []
    blacken_eyes(story.anjali, story.ghost_values)
    for material in story.anjali.mesh.data.materials:
        if material.name != "AnjaliEye":
            ghostify(material, story.ghost_values)

    build_set(story)
    story.shoe = extract_shoe(story.anjali, story.grave_position + Vector((0.05, 0.0, 0.13)), get_collection("Props"))
    build_end_cards(get_collection("Cards"))
    direct_people(story)
    direct_props_and_light(story)
    shots = build_shots(story)
    build_compositor(scene)
    clips.cleanup()

    # Source models are only instanced, so they stay out of the view layer.
    for layer_collection in bpy.context.view_layer.layer_collection.children:
        if layer_collection.name == "_Sources":
            layer_collection.exclude = True

    if take_poster_photos(story, force=rebuild_photos):
        paint_posters()
        for image in bpy.data.images:
            if "poster_" in image.filepath:
                image.reload()
    scene.frame_set(1)
    log(f"film built in {time.time() - started:.0f}s: {len(shots.shots)} shots, {len(bpy.data.objects)} objects")
    return story, shots


def save_film():
    write_credits()
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH), compress=True)
    log(f"saved {BLEND_PATH} ({BLEND_PATH.stat().st_size / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def enable_best_device(scene):
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            preferences.compute_device_type = kind
        except TypeError:
            continue
        preferences.get_devices()
        usable = [d for d in preferences.devices if d.type == kind]
        if usable:
            for device in preferences.devices:
                device.use = device.type == kind
            scene.cycles.device = "GPU"
            if hasattr(scene.cycles, "denoising_use_gpu"):
                scene.cycles.denoising_use_gpu = True
            log(f"rendering on {kind}: {[d.name for d in usable]}")
            return kind
    preferences.compute_device_type = "NONE"
    scene.cycles.device = "CPU"
    log("no supported GPU found, rendering on the CPU")
    return "CPU"


def render_test_frames(frames, scale, samples):
    scene = bpy.context.scene
    enable_best_device(scene)
    TEST_DIR.mkdir(exist_ok=True)
    scene.render.resolution_percentage = scale
    if samples:
        scene.cycles.samples = samples
    scene.render.use_persistent_data = False
    for frame in frames:
        scene.frame_set(frame)
        scene.render.filepath = str(TEST_DIR / f"frame_{frame:04d}.png")
        started = time.time()
        bpy.ops.render.render(write_still=True)
        log(f"frame {frame} ({scene.camera.name}) took {time.time() - started:.0f}s -> {scene.render.filepath}")


def render_final():
    scene = bpy.context.scene
    device = enable_best_device(scene)
    FRAMES_DIR.mkdir(exist_ok=True)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(FRAMES_DIR / "frame_")
    scene.render.use_overwrite = False     # finished frames are skipped, so it can be stopped and resumed
    scene.render.use_placeholder = False
    scene.frame_start, scene.frame_end = 1, LAST_FRAME
    done = len(list(FRAMES_DIR.glob("frame_*.png")))
    log(f"{done}/{LAST_FRAME} frames already rendered, continuing on {device}")
    bpy.ops.render.render(animation=True)
    log("all frames rendered")


def encode_video():
    scene = bpy.context.scene
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"No frames found in {FRAMES_DIR}")
    editor = scene.sequence_editor_create()
    strip = editor.strips.new_image("Film", str(frames[0]), 1, 1)
    for frame in frames[1:]:
        strip.elements.append(frame.name)
    scene.frame_start, scene.frame_end = 1, len(frames)
    render = scene.render
    render.use_sequencer = True
    render.use_compositing = False
    render.resolution_percentage = 100
    render.image_settings.media_type = "VIDEO"
    render.image_settings.file_format = "FFMPEG"
    render.ffmpeg.format = "MPEG4"
    render.ffmpeg.codec = "H264"
    render.ffmpeg.constant_rate_factor = "HIGH"
    render.ffmpeg.ffmpeg_preset = "GOOD"
    render.ffmpeg.audio_codec = "NONE"
    render.filepath = str(PROJECT / "last_seen.mp4")
    bpy.ops.render.render(animation=True)
    log(f"video written to {PROJECT / 'last_seen.mp4'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    def value(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    frames = value("--frames")
    return {
        "fetch_only": "--fetch-only" in argv,
        "final": "--final" in argv,
        "encode": "--encode" in argv,
        "photos": "--new-photos" in argv,
        "frames": [int(f) for f in frames.split(",")] if frames else None,
        "scale": int(value("--scale", 50)),
        "samples": int(value("--samples", 0)),
    }


def main():
    opts = parse_args()
    already_built = bool(bpy.data.filepath) and Path(bpy.data.filepath).name == BLEND_PATH.name
    if opts["encode"]:
        encode_video()
        return
    if opts["final"]:
        render_final()
        return
    if not already_built:
        fetch_all_assets()
        if opts["fetch_only"]:
            return
        build_film(rebuild_photos=opts["photos"])
        save_film()
    if opts["frames"] or not already_built:
        render_test_frames(opts["frames"] or [DEFAULT_TEST_FRAME], opts["scale"], opts["samples"])


if __name__ == "__main__":
    main()
