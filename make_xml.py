# _*_ coding: utf-8 _*_
import os
import glob
import json

import xmltodict


def make_xml(video_dir: str) -> None:

    xml_dir = os.path.join(video_dir, "xml")
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)

    with open(os.path.join(video_dir, "result.json"), "r") as f:
        info = json.load(f)

    for frame_name, images in info.items():
        xml_dict = {
            "annotation": {
                "folder": os.path.basename(video_dir),
                "filename": frame_name,
                "path": os.path.join(video_dir, "images", frame_name),
                "source": {
                    "database": "Unknown",
                },
                "object": list(),
            },
        }
        for person_id, image in images.items():
            xml_dict["annotation"]["object"].append({
                "name": "person",
                "pose": "Unspecified",
                "truncated": "0",
                "difficult": "0",
                "bndbox": {
                    "xmin": image["left"],
                    "ymin": image["top"],
                    "xmax": image["right"],
                    "ymax": image["bottom"],
                },
            })

        with open(os.path.join(xml_dir, os.path.splitext(frame_name)[0] + ".xml"), "w") as f:
            f.write(xmltodict.unparse(xml_dict, pretty=True))


def make_xml_multi(base_dir: str) -> None:
    video_dirs = glob.glob(os.path.join(base_dir, "*"))
    for video_dir in video_dirs:
        make_xml(video_dir)


if __name__ == "__main__":
    BASE_DIR = "/home/hosokawa/share/satsudora/crop"
    make_xml_multi(BASE_DIR)


# vim:set fenc=utf-8 ff=unix expandtab sw=4 ts=4:
