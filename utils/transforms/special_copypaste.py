import random
import os.path as osp
import json
import warnings
from typing import List, Tuple, Union

import cv2
from pycocotools import mask
import numpy as np
import math
import torch

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks
from mmdet.structures.bbox import HorizontalBoxes

class PasteObject(object):
    def __init__(self, object_info, category, category_id, img_dir) -> None:
        self.image_id = object_info['image_id']
        self.category = category
        self.category_id = category_id

        self.paste_coord = None

        # load image
        self.img = cv2.imread(osp.join(img_dir, object_info['file_name']))

        # load binary mask
        self.mask = mask.decode(object_info['segmentation']).astype(np.uint8)

        self.size = (self.mask.shape[1], self.mask.shape[0])
        
@TRANSFORMS.register_module()
class SpecialCopyPaste(BaseTransform):
    """Specialized CopyPaste Augmentation. It will read the cropped objects and their mask annotation from dataset
    then randomly paste them on the source image.
    For MMSports 2023 competition, we also implement a basketball court estimator to estimate pasting area.
    This technique will help augmented image look more realistic

    Args:
        crop_dir (str): the folder dir where cropped objects are stored
        crop_anno (str): the name of annotation file
        max_num_objects (list[int]): the maximum number of pasted objects for each class.
        bbox_occluded_thr, mask_occluded_thr (int): thresholds to filter the totally occluded objects after pasting
    """

    def __init__(
        self, 
        crop_dir: str, 
        crop_anno: str, 
        max_num_objects: List[int],
        bbox_occluded_thr: int = 10,
        mask_occluded_thr: int = 300,
        prob: float = 0.5,
        occl_prob: float = 0.5
        ) -> None:
        assert 0 <= prob <= 1
        assert 0 <= occl_prob <= 1
        
        self.crop_dir = crop_dir
        self.crop_anno = crop_anno
        self.mask_info = dict()
        self.categories = []
        self.categorie_ids = []
        self.max_num_objects = max_num_objects
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr

        self.prob = prob
        self.occl_prob = occl_prob

        # The minimum contour area to be considered in the basketball court detector. 
        # It should be big to detect just the court and discard the small objects.
        self.min_contour_area = 100000

        # Minimum line lenght after Hough line detector. 
        # We only want to take large lines representing the edges of the basketball court, not short lines.
        self.min_line_length = 300

        # Minimum angle to be considered a valid intersection. 
        # If the angle between two lines is lower than that number it won't be considered as an intersection,
        # as the lines would be too similar.
        self.min_angle = 25

        # The distnce at which the intersections are averaged. 
        # If two intersections are closer than that number, they will be averaged into one point.
        self.min_point_distance = 80 
        
        self.paste_list: List[PasteObject] = []
        
        # prepare mask info
        self._load_object_list()
        assert len(self.categories) == len(self.max_num_objects)
        

    def _random_prob(self) -> float:
        return random.uniform(0, 1)

    def _load_object_list(self):
        '''Load categories from crop annotation file then export all info into a dict'''
        crop_anno_path = osp.join(self.crop_dir, self.crop_anno)

        with open(crop_anno_path, 'r') as json_file:
            json_anno = json.load(json_file)

        for i, cat in enumerate(json_anno['categories']):
            self.categories.append(cat['name'])
            self.categorie_ids.append(i)
            self.mask_info[cat['name']] = json_anno[cat['name']]

    def _select_object(self):
        '''select objects randomly from crop_anno to paste to image'''
        # clear selected objects from last augmentation
        self.paste_list = []

        for i in range(len(self.categories)):
            cat = self.categories[i]
            cat_id = self.categorie_ids[i]

            # image dir
            img_dir = osp.join(self.crop_dir, cat)

            candidates = self.mask_info[cat]

            # random number of paste objecs
            paste_num = random.randint(0, self.max_num_objects[i])

            # randomly sample (duplicate is allowed)
            samples = random.choices(candidates, k=paste_num)

            # add to paste list
            self.paste_list = self.paste_list + \
                [PasteObject(info, cat, cat_id, img_dir) for info in samples]
            
        # shuffle the list
        random.shuffle(self.paste_list)

    def _average_close_points(self, points, threshold):
        '''averaging the intersections for defining the pasting area
        
        Args:
        - points: A list with all the intersection points.
        - threshold: A threshold to determine the minimum distance at which intersections can be.

        Returns:
        - a list of points that are too close to each other averaged into one
        '''
        averaged_points = []
        while len(points) > 0:
            current_point = points[0]
            close_points = [current_point]
            remaining_points = []

            for point in points[1:]:
                distance = np.linalg.norm(np.array(current_point) - np.array(point))
                if distance <= threshold:
                    close_points.append(point)
                else:
                    remaining_points.append(point)

            averaged_point = np.mean(np.array(close_points), axis=0)
            averaged_points.append(averaged_point.tolist())
            points = remaining_points

        return averaged_points

    def _assign_paste_coord(self, results):
        '''assign paste coord for each paste object'''
        # Read the image and get its dimensions
        image = results['img']
        h, w = results['img_shape']
        
        # Check if the image is right or left side of the court
        image_name = osp.splitext(osp.basename(results['img_path']))[0]
        if image_name.startswith('camcourt1'):
            right = 0
        else:
            right = 1
        
        # Threshold the image
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresholded = cv2.threshold(image_gray, 120, 150, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Contour the thresholded image and filter the contours to get only the court
        filtered_contours = []
        blank_image = np.zeros_like(image)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:  # Adjust the threshold as needed
                filtered_contours.append(contour)
        cv2.drawContours(blank_image, filtered_contours, -1, (0, 255, 0), 2)
        
        # Apply Hough Line Transform to detect straight lines
        gray = cv2.cvtColor(blank_image, cv2.COLOR_RGB2GRAY)
        lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

        # Draw the detected lines on the original image
        blank_image_2 = np.zeros_like(image)
        enlarged_lines = []
        scale_factor = 3.5
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length > self.min_line_length:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    dx = x2 - x1
                    dy = y2 - y1

                    # Compute the new endpoint coordinates
                    new_x1 = center_x - int(dx * scale_factor / 2)
                    new_y1 = center_y - int(dy * scale_factor / 2)
                    new_x2 = center_x + int(dx * scale_factor / 2)
                    new_y2 = center_y + int(dy * scale_factor / 2)

                    # Append the enlarged line to the list of enlarged lines
                    enlarged_lines.append([[new_x1, new_y1, new_x2, new_y2]])
                    
        for line in enlarged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(blank_image_2, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Find intersections of line segments
        filtered_intersections = []
        if enlarged_lines is not None:
            for i in range(len(enlarged_lines)):
                for j in range(i+1, len(enlarged_lines)):
                    x1, y1, x2, y2 = enlarged_lines[i][0]
                    x3, y3, x4, y4 = enlarged_lines[j][0]

                    if x1 != x2 and x3 != x4: #No lines are vertical
                        m1 = (y2-y1)/(x2-x1)
                        m2 = (y4-y3)/(x4-x3)
                        if m1 != m2: #The lines intersect
                            intersect_x = ((m1 * x1) - (m2 * x3) + y3 - y1) / (m1 - m2)
                            intersect_y = m1 * (intersect_x - x1) + y1
                        else: #There is no intersection
                            intersect_x = None
                            intersect_y = None
                    elif x1 == x2 and x3 != x4: #Line 1 only is vertical
                        m2 = (y4/y3)/(x4-x3)
                        intersect_x = x1
                        intersect_y = y3 - m2*(intersect_x - x3)
                    
                    elif x3 == x4 and x1 != x2: #Line 2 only is vertical
                        m1 = (y2/y1)/(x2-x1)
                        intersect_x = x3
                        intersect_y = y1 - m1*(intersect_x - x1)

                    elif x1==x2 and x3==x4: #Both lines are vertical
                        intersect_x = None
                        intersect_y = None
                    
                    elif y1==y2 and y3==y4: #Both lines are horizontal
                        intersect_x = None
                        intersect_y = None
                    
                    if intersect_x != None and intersect_y != None: #The intersection exists
                        if x1 < intersect_x and intersect_x < x2 and y1 < intersect_y and intersect_y < y2: #the intersection is inside the lines
                            
                            # Get the direction vectors of the lines
                            v1 = (x2 - x1, y2 - y1)
                            v2 = (x4 - x3, y4 - y3)
                            
                            # Calculate the magnitudes of the direction vectors
                            v1_magnitude = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
                            v2_magnitude = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
                            
                            # Normalize the direction vectors
                            v1_normalized = (v1[0] / v1_magnitude, v1[1] / v1_magnitude)
                            v2_normalized = (v2[0] / v2_magnitude, v2[1] / v2_magnitude)

                            # Calculate the dot product of the normalized vectors
                            dot_product = v1_normalized[0] * v2_normalized[0] + v1_normalized[1] * v2_normalized[1]

                            # Calculate the intersection angle in radians
                            angle_radians = np.arccos(dot_product)

                            # Convert the angle to degrees
                            angle_degrees = np.degrees(angle_radians)
                            if angle_degrees > self.min_angle:
                                filtered_intersections.append((intersect_x, intersect_y))
                    
        filtered_intersections = self._average_close_points(filtered_intersections, self.min_point_distance)
        
        for point in filtered_intersections:
            intersect_x, intersect_y = point
            cv2.circle(blank_image_2, (int(intersect_x), int(intersect_y)), radius=10, color=(0, 0, 255), thickness=-1)
        
        gray = cv2.cvtColor(blank_image_2, cv2.COLOR_RGB2GRAY)
        # Apply Hough Line Transform to detect straight lines, to get the relative coords of the enlarged lines
        new_lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

        if new_lines is None:
            new_lines = []
            
        blank_image_3 = np.zeros_like(image)
        for line in new_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(blank_image_3, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get the points most to the right or left 
        extreme_points = []
        for line in new_lines:
            x1, y1, x2, y2 = line[0]
            if right == 1:
                if x1 == 0:
                    extreme_points.append((x1,y1))
            else:
                if x2 == w-1:
                    extreme_points.append((x2,y2))
        
        # Compare the points to get the topmost and bottomost
        topmost_point = None
        bottommost_point = None
        for point in extreme_points:
            x, y = point
            if topmost_point is None or y < topmost_point[1]:
                topmost_point = (x, y)
            if bottommost_point is None or y > bottommost_point[1]:
                bottommost_point = (x, y)
        
        # Apply thresholds on the two points if they exist
        if topmost_point != None and bottommost_point != None:
            x, y = topmost_point
            if y < h//3:
                topmost_point = (x, h//3)
            elif y > h//2:
                topmost_point = (x, h//2)
            x, y = bottommost_point
            if y > 5*h//6:
                bottommost_point = (x, 5*h//6)
            elif y < 3*h//4:
                bottommost_point = (x, 3*h//4)
        
            cv2.circle(blank_image_2, topmost_point, radius=10, color=(0, 0, 255), thickness=-1)
            cv2.circle(blank_image_2, bottommost_point, radius=10, color=(0, 0, 255), thickness=-1)
        
        # Get top and bot intersections
        top_inter = None
        bot_inter = None
        for point in filtered_intersections:
            x, y = point
            if top_inter is None or y < top_inter[1]:
                top_inter = (x, y)
            if bot_inter is None or y > bot_inter[1]:
                bot_inter = (x, y)
        
        # In case any point is not detected, apply predefined area
        apply_predefined = False
        if top_inter == None or bot_inter == None or topmost_point == None or bottommost_point == None or top_inter == bot_inter or top_inter == topmost_point or bot_inter == bottommost_point or topmost_point == bottommost_point:
            apply_predefined = True
            if right == 1:
                topmost_point = (0, h//2 - h//5)
                top_inter = (w - w//5, h//2 - h//5)
                bot_inter = (w - w//5, h//2 + h//5)
                bottommost_point = (0, h//2 + h//5)
            else:
                top_inter = (w//5, h//2 - h//5)
                topmost_point = (w, h//2 - h//5)
                bottommost_point = (w, h//2 + h//5)
                bot_inter = (w//5, h//2 + h//5)
        
        if right == 1:
            final_points = [topmost_point, top_inter, bot_inter, bottommost_point]
            min_x = 0
            max_x = bot_inter[0]
            min_y = bottommost_point[1]
            max_y = top_inter[1]
        else:
            final_points = [top_inter, topmost_point, bottommost_point, bot_inter]
            min_x = bot_inter[0]
            max_x = w
            min_y = bottommost_point[1]
            max_y = top_inter[1]
        
        final_points = np.array(final_points, dtype=np.int32)
        # cv2.polylines(image, [np.array(final_points)], isClosed=True, color=(255, 0, 0), thickness=6)    
        
        # Create pasting area
        hull = cv2.convexHull(final_points)

        i = 0
        # for paste_object in self.paste_list:
        while i < len(self.paste_list):
            paste_object = self.paste_list[i]
            point_inside = False
            while not point_inside:
                random_x = random.uniform(min_x, max_x)
                random_y = random.uniform(min_y, max_y)
                distance = cv2.pointPolygonTest(hull, (random_x, random_y), measureDist=False)
                if distance > 0:
                    point_inside = True
            
            # calculate paste coordinate
            paste_w, paste_h = paste_object.size

            # (random_x, random_y) should be bottom-right if basketball court can be detected
            # otherwise, it should be top-left
            if apply_predefined:
                paste_x = random_x
                paste_y = random_y
            else:
                paste_x = random_x - paste_w
                paste_y = random_y - paste_h

            # validate paste coordinate
            paste_x = math.floor(max(min(paste_x, w-paste_w), 0.0))
            paste_y = math.floor(max(min(paste_y, h-paste_h), 0.0))

            paste_object.paste_coord = (paste_x, paste_y)

            # update mask to image scale
            expanded_mask = np.zeros(shape=(h, w), dtype=np.uint8)
            expanded_mask[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = paste_object.mask
            paste_object.mask = expanded_mask

            # create occlusion
            if i < len(self.paste_list) - 1:
                next_paste_object = self.paste_list[i+1]
                next_paste_w, next_paste_h = next_paste_object.size

                if self._random_prob() <= self.occl_prob:
                    next_paste_x = math.floor(random.uniform(paste_x, paste_x+paste_w))
                    next_paste_y = math.floor(random.uniform(paste_y, paste_y+paste_h))

                    # validate paste coordinate
                    next_paste_x = math.floor(max(min(next_paste_x, w-next_paste_w), 0.0))
                    next_paste_y = math.floor(max(min(next_paste_y, h-next_paste_h), 0.0))

                    next_paste_object.paste_coord = (next_paste_x, next_paste_y)

                    next_expanded_mask = np.zeros(shape=(h, w), dtype=np.uint8)
                    next_expanded_mask[next_paste_y:next_paste_y+next_paste_h, next_paste_x:next_paste_x+next_paste_w] = next_paste_object.mask
                    next_paste_object.mask = next_expanded_mask

                    i += 1

            # increase index
            i += 1

    def _update_occluded_masks(self):
        '''update mask of paste objects if occluded'''
        paste_masks = np.array([paste_object.mask for paste_object in self.paste_list])

        for i, paste_object in enumerate(self.paste_list):
            if i == len(self.paste_list) - 1:
                break

            composed_mask = np.where(np.any(paste_masks[i+1:], axis=0), 1, 0)
            paste_object.mask[composed_mask == 1] = 0
        
    def _get_updated_masks(self, masks: BitmapMasks,
                           composed_mask: np.ndarray) -> BitmapMasks:
        """Update masks with composed mask."""
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks
    
    def _paste_list_to_gt_bboxes(self, dtype, device):
        '''Convert paste list to goundtruth bboxes by mmdet format'''
        bboxes = []

        for paste_object in self.paste_list:
            x_top, y_top = paste_object.paste_coord
            w, h = paste_object.size
            bboxes.append(np.array([x_top+w/2, y_top+h/2, w, h]))

        return HorizontalBoxes(
            data=np.array(bboxes),
            dtype=dtype, 
            device=device,
            in_mode='cxcywh'
        )
    
    def _paste_list_to_gt_masks(self):
        '''Convert paste list to goundtruth masks by mmdet format'''
        paste_masks = np.array([paste_object.mask for paste_object in self.paste_list])

        return BitmapMasks(
            masks=paste_masks,
            height=paste_masks.shape[1],
            width=paste_masks.shape[2]
        )
    
    def _paste_list_to_gt_labels(self):
        '''Convert paste list to goundtruth label by mmdet format'''
        return np.array([paste_object.category_id for paste_object in self.paste_list])
    
    def _paste_list_to_gt_ignore_flags(self):
        '''Convert paste list to goundtruth ignore flags by mmdet format'''
        return np.full((len(self.paste_list),), False, dtype=bool)

    def transform(self, results):
        if self._random_prob() > self.prob:
            return results
          
        self._select_object()
        self._assign_paste_coord(results)
        self._update_occluded_masks()

        # paste all selected objects to img
        dst_img = results['img']
        dst_bboxes = results['gt_bboxes']
        dst_labels = results['gt_bboxes_labels']
        dst_masks = results['gt_masks']
        dst_ignore_flags = results['gt_ignore_flags']

        if len(self.paste_list) == 0:
            return results
        
        # get paste ground truth
        src_bboxes = self._paste_list_to_gt_bboxes(
            dtype=dst_bboxes.dtype, 
            device=dst_bboxes.device)
        src_masks = self._paste_list_to_gt_masks()
        src_labels = self._paste_list_to_gt_labels()
        src_ignore_flags = self._paste_list_to_gt_ignore_flags()

        if dst_masks is not None:
            # update masks and generate bboxes from updated masks
            composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
            updated_dst_masks = self._get_updated_masks(dst_masks, composed_mask)
            updated_dst_bboxes = updated_dst_masks.get_bboxes(type(dst_bboxes))
            assert len(updated_dst_bboxes) == len(updated_dst_masks)

            # filter totally occluded objects
            l1_distance = (updated_dst_bboxes.tensor - dst_bboxes.tensor).abs()
            bboxes_inds = (l1_distance <= self.bbox_occluded_thr).all(
                dim=-1).numpy()
            masks_inds = updated_dst_masks.masks.sum(
                axis=(1, 2)) > self.mask_occluded_thr
            valid_inds = bboxes_inds | masks_inds

            # update gt info
            bboxes = src_bboxes.cat([updated_dst_bboxes[valid_inds], src_bboxes])
            labels = np.concatenate([dst_labels[valid_inds], src_labels])
            masks = np.concatenate(
                [updated_dst_masks.masks[valid_inds], src_masks.masks])
            ignore_flags = np.concatenate(
                [dst_ignore_flags[valid_inds], src_ignore_flags])
        else:
            bboxes = src_bboxes
            labels = src_labels
            masks = src_masks.masks
            ignore_flags = src_ignore_flags

        # paste objects onto image directly
        img = dst_img
        for paste_object in self.paste_list:
            x_top, y_top = paste_object.paste_coord
            w, h = paste_object.size
            cropped_mask = paste_object.mask[y_top:y_top+h, x_top:x_top+w]
            img[y_top:y_top+h, x_top:x_top+w][cropped_mask==1] = paste_object.img[cropped_mask==1]
            
        results['img'] = img
        results['gt_bboxes'] = bboxes
        results['gt_bboxes_labels'] = labels
        results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                              masks.shape[2])
        results['gt_ignore_flags'] = ignore_flags

        return results
