"""
CLI tool for downloading images for a given area
"""

import os
import json
import shutil
import argparse
import numpy as np

from geemap.common import  get_image_thumbnail
from geemap.timelapse import landsat_timeseries
import ee


def fetch_images(roi, target_path, start_year=1984, end_year=2021, dimensions=1800,  frequency='year', start_img_index=0):
    start_date = "06-10"
    end_date = "09-20"
    with open(os.path.join(target_path, f'metadata_{start_img_index}.json'), 'w') as f:
        json.dump({
            "roi" : None, # for now,
            "start_year": start_year,
            "end_year": end_year,
            "start_date": start_date,
            "end_date": end_date,
            "dimension": dimensions,
            "frequency": frequency,
            "start_img_index": start_img_index
        }, f)

    bands = ["Red", "Green", "Blue"]

    allowed_bands = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "pixel_qa"]

    if len(bands) == 3 and all(x in allowed_bands for x in bands):
        pass
    else:
        raise Exception(
            "You can only select 3 bands from the following: {}".format(
                ", ".join(allowed_bands)
            )
        )

    vis_params = {}
    vis_params["bands"] = bands
    vis_params["min"] = 0
    vis_params["max"] = 4000
    vis_params["gamma"] = [1, 1, 1]

    col = landsat_timeseries(
        roi,
        start_year,
        end_year,
        start_date,
        end_date,
        frequency=frequency
    )
    col = col.select(bands).map(
        lambda img: img.visualize(**vis_params).set(
            {
                "system:time_start": img.get("system:time_start"),
                "system:date": img.get("system:date"),
            }
        )
    )

    ee_object = col
    vis_params = {
        "min": 0,
        "max": 255,
        "bands": ["vis-red", "vis-green", "vis-blue"],
        "dimensions": dimensions
    }

    count = int(ee_object.size().getInfo())
    images = ee_object.toList(count)

    for i in range(0, count):
        image = ee.Image(images.get(i))
        if i < start_img_index:
            continue
        out_img = os.path.join(target_path, f'img_{i}.jpg')
        print(f'Downloading {i + 1}/{count}')

        get_image_thumbnail(image, out_img, vis_params, dimensions)


if __name__ == "__main__":
    ee.Initialize()
    parser = argparse.ArgumentParser()

    parser.add_argument('--area', type=str, required=True,
                        help='Area for which we want to fetch images')
    parser.add_argument('--frequency', type=str, default='year',
                        help='Frequency of the pictures, month/quarter/year')
    parser.add_argument('--dimensions', type=int, default=1800,
                        help='Dimension of the image in px')
    args = parser.parse_args()

    with open(os.path.join(args.area), 'r') as f:
        roi_coords = json.load(f)
    roi = ee.Geometry.Polygon(roi_coords, None, False)

    out_dir = os.path.join('/'.join(os.path.split(args.area)[0].split('/')[:-1]),
                           os.path.split(args.area)[1].split('.')[0])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fetch_images(roi=roi, target_path=out_dir, frequency=args.frequency, dimensions=args.dimensions)