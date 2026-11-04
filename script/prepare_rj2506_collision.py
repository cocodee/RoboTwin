"""Build convex finger collision parts without filling RJ2506's gripping slots.

Requires the original RJ2506 meshes/URDF and coacd==1.0.14 (generation only).
The generated assets are included in the repository; normal runs need no CoACD.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import coacd
import trimesh


def prepare(root):
    source = root / "urdf/relpath_RJ2506_fixed.urdf"
    destination = root / "urdf/relpath_RJ2506_compact.urdf"
    output = root / "meshes/convex_compact"
    output.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    for side in ("left", "right"):
        for finger in (1, 2):
            name = f"{side}_hand_finger{finger}"
            mesh = trimesh.load(root / "meshes" / f"{name}.STL")
            parts = coacd.run_coacd(
                coacd.Mesh(mesh.vertices, mesh.faces),
                threshold=0.05,
                max_convex_hull=16,
                preprocess_mode="off",
                seed=0,
            )
            link = tree.find(f"./link[@name='{name}']")
            if link is None:
                raise ValueError(f"Missing finger link: {name}")
            for collision in link.findall("collision"):
                link.remove(collision)
            for index, (vertices, faces) in enumerate(parts):
                filename = f"{name}_{index}.obj"
                trimesh.Trimesh(vertices, faces).export(output / filename)
                geometry = ET.SubElement(ET.SubElement(link, "collision"), "geometry")
                ET.SubElement(geometry, "mesh", filename=f"../meshes/convex_compact/{filename}")
            print(f"{name}: {len(parts)} convex parts")
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-dir", type=Path, default=Path("assets/embodiments/RJ2506"))
    prepare(parser.parse_args().robot_dir)
