"""Decode a GLB's triangle meshes to OBJ without third-party Python packages.

Designed for this repository's pinned Hitmonlee reference. Preserves per-primitive
mesh groups but intentionally leaves materials/textures in the original GLB.
"""
from __future__ import annotations
import json
import struct
import argparse
from pathlib import Path

COMPONENTS = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path):
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("Invalid or unsupported GLB 2.0 file")
    offset, document, binary = 12, None, None
    while offset < len(data):
        length, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        offset += length
        if kind == b"JSON":
            document = json.loads(chunk)
        elif kind == b"BIN\0":
            binary = chunk
    if document is None or binary is None:
        raise ValueError("GLB must contain JSON and BIN chunks")
    return document, binary


def accessor(doc, blob, index):
    spec = doc["accessors"][index]
    view = doc["bufferViews"][spec["bufferView"]]
    fmt, component_size = COMPONENTS[spec["componentType"]]
    width = WIDTHS[spec["type"]]
    stride = view.get("byteStride", component_size * width)
    start = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
    values = []
    for row in range(spec["count"]):
        chunk = blob[start + row * stride:start + row * stride + component_size * width]
        values.append(struct.unpack("<" + fmt * width, chunk))
    return values


def reference_transform(point, transform):
    """Map source Y-up, +Z-facing coordinates into robot X-forward/Z-up mm."""
    if transform is None:
        return point
    x, y, z = point
    s = transform["scale_mm_per_source_unit"]
    return (
        (z - transform["source_depth_center"]) * s,
        x * s,
        (y - transform["source_vertical_min"]) * s,
    )


def main(source: Path, destination: Path, design_config: Path | None = None):
    doc, blob = read_glb(source)
    transform = None
    if design_config is not None:
        config = json.loads(design_config.read_text(encoding="utf-8"))
        reference = config["appearance_reference"]
        min_y = min(a["min"][1] for a in doc["accessors"] if "min" in a and a["type"] == "VEC3")
        max_y = max(a["max"][1] for a in doc["accessors"] if "max" in a and a["type"] == "VEC3")
        depth_mins = [a["min"][2] for a in doc["accessors"] if "min" in a and a["type"] == "VEC3"]
        depth_maxs = [a["max"][2] for a in doc["accessors"] if "max" in a and a["type"] == "VEC3"]
        target_height = reference["normalized_height_mm"]
        transform = {
            "source_vertical_min": min_y,
            "scale_mm_per_source_unit": target_height / (max_y - min_y),
            "source_depth_center": (min(depth_mins) + max(depth_maxs)) / 2,
        }
    mesh_names = []
    with destination.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# Mesh reference decoded from pinned source GLB.\n")
        out.write("# Keep original GLB for textures and complete source metadata.\n")
        if transform:
            out.write("# Transformed into robot frame and normalized to configured reference height.\n")
        vertex_base = 1
        for mesh in doc["meshes"]:
            for primitive_index, primitive in enumerate(mesh["primitives"]):
                positions = accessor(doc, blob, primitive["attributes"]["POSITION"])
                indices = accessor(doc, blob, primitive["indices"])
                name = mesh.get("name", "mesh") + (f"_{primitive_index}" if len(mesh["primitives"]) > 1 else "")
                mesh_names.append((name, len(positions), len(indices)))
                out.write(f"\ng {name}\nusemtl {doc['materials'][primitive['material']].get('name', 'default')}\n")
                for xyz in positions:
                    out.write("v %.9g %.9g %.9g\n" % reference_transform(xyz, transform))
                flat_indices = [row[0] for row in indices]
                if len(flat_indices) % 3:
                    raise ValueError(f"Non-triangle index count in {name}")
                for i in range(0, len(flat_indices), 3):
                    a, b, c = (vertex_base + flat_indices[j] for j in (i, i + 1, i + 2))
                    out.write(f"f {a} {b} {c}\n")
                vertex_base += len(positions)
    summary = {
        "source": str(source),
        "output": str(destination),
        "mesh_groups": [{"name": n, "vertices": v, "indices": i, "triangles": i // 3} for n, v, i in mesh_names],
        "total_vertices": sum(v for _, v, _ in mesh_names),
        "total_triangles": sum(i // 3 for _, _, i in mesh_names),
        "materials_and_textures": "preserved in source GLB; OBJ is geometry reference only",
        "robot_frame_transform": transform,
    }
    destination.with_suffix(".conversion.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--design-config", type=Path, help="Apply the frame and scale in a CAD design JSON")
    args = parser.parse_args()
    main(args.source, args.destination, args.design_config)
