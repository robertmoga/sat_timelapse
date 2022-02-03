"""
Pipeline for postprocess images.

Details about steps
"""

import os
import json
import tqdm
import shutil
import logging

import numpy as np
import matplotlib.pyplot as plt

import cv2
from natsort import natsorted
from datetime import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WHITES_THRESHOLD = 200
BLACKS_THRESHOLD = 10

MAX_FAULTY_PX = 10000

REEL_RATIO = 0.5652


def fill_in_image_pixels(img_path, image, thresholded, img_inventory):
    """
    Fill in black pixels in images which might occur as error at acquisition time
    :param img_path:
    :param image:
    :param thresholded:
    :return:
    """
    img = image.copy()
    black_px_indexes = {x: None for x in zip(np.where(thresholded == 0)[0], np.where(thresholded == 0)[1])}
    current_img_index = int(os.path.split(img_path)[1].split('_')[-1].split('.')[0])

    finished = False
    for k in range(current_img_index + 1, len(img_inventory)):
        current_img = cv2.imread(os.path.join(os.path.split(img_path)[0], f'img_{k}.jpg'), 1)
        current_gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        for index_0, index_1 in black_px_indexes:
            if current_gray_img[index_0, index_1]:
                black_px_indexes[(index_0, index_1)] = current_img[index_0, index_1]
        finished = True if not len([v for k, v in black_px_indexes.items() if v is None]) else False

        if finished:
            break
    if not finished:
        # look for previous images
        for k in range(current_img_index, 0, -1):
            current_img = cv2.imread(os.path.join(os.path.split(img_path)[0], f'img_{k}.jpg'), 1)
            current_gray_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            for index_0, index_1 in black_px_indexes:
                if current_gray_img[index_0, index_1]:
                    black_px_indexes[(index_0, index_1)] = current_img[index_0, index_1]
            finished = True if not len([v for k, v in black_px_indexes.items() if v is None]) else False

            if finished:
                break

    for (index_0, index_1), px_value in black_px_indexes.items():
        img[index_0, index_1] = np.array(px_value)

    return img


def sharpen_image(image):
    img = image.copy()

    # add param to switch between them
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # adds to much structure to the clouds
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    img = cv2.filter2D(img, -1, kernel)
    return img


def equalize_histogram(image):
    img = image.copy()

    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))
    return equ


def process_images(img_inventory):
    logger.debug('Started image processing_images')

    with open('config.json', 'r') as f:
        config = json.load(f)

    valid_images = {img_name: img_details for img_name, img_details in img_inventory.items() if img_details['keep']}

    target_path = os.path.join(img_dir_path, 'processed_imgs')
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)

    for i, (fname, img_details) in tqdm.tqdm(enumerate(valid_images.items())):
        img = cv2.imread(os.path.join(img_dir_path, fname), 1)

        # fill in black pixels
        if config['perform_pixel_fill_in']:
            # check if frame needs filling
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(gray_img, BLACKS_THRESHOLD, 255, cv2.THRESH_BINARY)

            # fill frame
            if len(thresh1[thresh1 == 0]):
                img = fill_in_image_pixels(img_path=os.path.join(img_dir_path, fname),
                                           image=img,
                                           thresholded=thresh1,
                                           img_inventory=img_inventory)
                img_details['filled_pixels'] = True

        # sharpening
        if config['perform_sharpening']:
            img = sharpen_image(img)
            img_details['sharpened'] = True

        # histogram equalization
        if config['perform_histogram_equalization']:
            img = equalize_histogram(img)
            img_details['hist_equalization'] = True

        if config['resize']:
            img = cv2.resize(img, (img_details['dimensions'][1], int(img_details['dimensions'][1] * REEL_RATIO)), interpolation=cv2.INTER_AREA)

        if config['rotate']: # TODO automate rotation decesion making
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_details['dimensions'] = img.shape
            text_offset = 0.05
        else:
            text_offset = 0.1
        # add date
        cv2.putText(img, str(img_details['year']),
                    (50, int(img_details['dimensions'][0] - (img_details['dimensions'][0] * text_offset + 50))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, 'Cluj-Napoca',
                    (50, int(img_details['dimensions'][0] - img_details['dimensions'][0] * text_offset)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        # add progressbar
        if config['add_progress_bar']:
            img[img.shape[0]-20:img.shape[0], :int(img.shape[1]*(i/(len(valid_images)-1)))] = [199, 122, 8]

        cv2.imwrite(os.path.join(target_path, fname.split('.')[0] + '_prc.jpg'), img)
        img_details['processed_img_name'] = os.path.join(target_path, fname.split('.')[0] + '_prc.jpg')


def filter_images(img_dir_path):
    """
    Filter out the images that include processing errors or are overexposed.

    The means
    :return:
    """
    logger.info('Started image filtering')
    images_to_keep = {}
    for fname in tqdm.tqdm(natsorted(os.listdir(img_dir_path))):
        if not fname.endswith('.jpg'):
            continue

        img_gray = cv2.imread(os.path.join(img_dir_path, fname), 0)
        img_vals = img_gray.ravel()

        images_to_keep[fname] = {}
        images_to_keep[fname]['dimensions'] = img_gray.shape
        if img_vals[img_vals < BLACKS_THRESHOLD].shape[0] > MAX_FAULTY_PX or img_vals[img_vals > WHITES_THRESHOLD].shape[0] > MAX_FAULTY_PX:
            images_to_keep[fname]["keep"] = False
        else:
            images_to_keep[fname]["keep"] = True

    return images_to_keep


def define_dates(img_inventory):
    logger.info('Started defining dates')
    with open(os.path.join(img_dir_path, 'metadata_0.json'), 'r') as f:
        metadata = json.load(f)

    start_date = str(metadata['start_year']) + '-' + metadata['start_date']
    end_date = str(metadata['end_year']) + '-' + metadata['end_date']

    current_datetime = date_time_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = date_time_obj = datetime.strptime(end_date, '%Y-%m-%d')

    time_offset = {'year': relativedelta(years=+1),
                   'quarter': relativedelta(months=+3),
                   'month': relativedelta(months=+1)}

    for i, fname in enumerate(img_inventory):
        img_inventory[fname]['year'] = current_datetime.year
        current_datetime += time_offset[metadata['frequency']]

    return None


def build_video(img_dir_path, img_inventory):
    logger.info('Start video creation')
    first_frame_details = [img_details for img_name, img_details in img_inventory.items() if img_details['keep']][0]
    vid_w, vid_h = first_frame_details['dimensions'][1], first_frame_details['dimensions'][0]
    print(vid_w, vid_h)
    out_file = os.path.join(img_dir_path, f"{img_dir_path.split('/')[-1]}.mp4") # change this to not depend on the folder name
    out = cv2.VideoWriter(out_file,
                                cv2.VideoWriter_fourcc(*'MP4V'),
                                7,
                                (vid_w, vid_h))

    for img_name, img_details in tqdm.tqdm(img_inventory.items()):
        if img_details['keep']:
            img = cv2.imread(img_details['processed_img_name'], 1)
            out.write(img)

    last_image_details = [img_details for img_name, img_details in img_inventory.items() if img_details['keep']][-1]
    last_img = cv2.imread(last_image_details['processed_img_name'], 1)
    for i in range(15):
        out.write(last_img)

    out.release()
    print(f'Saved to {out_file}')


if __name__ == "__main__":
    img_dir_path = '/home/robert/dev/gif_tests/cluj_2'
    img_inventory = filter_images(img_dir_path)
    define_dates(img_inventory)
    process_images(img_inventory)
    build_video(img_dir_path, img_inventory)

# TODO : Fix logs