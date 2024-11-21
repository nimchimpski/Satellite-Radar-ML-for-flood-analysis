import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

from chainercv.evaluations import eval_semantic_segmentation
from PIL import Image
import cv2

from surface_distance.metrics import compute_surface_distances, compute_surface_dice_at_tolerance
from sklearn.metrics import confusion_matrix

Image.MAX_IMAGE_PIXELS = None


base = 'Y:\\Users\\Jiakun\\FloodAI\\scripts\\flood-59'

root = 'AI20211001THA'
sar = 'S1A_clip1.tif'
gt = 'gt.tif'
ai_output_1 = 'AI_output_postprocess.tif'
ai_output_2 = 'AI_output_wo_postprocess.tif'

# root = 'AI20200708NPL'
# sar = 'S1A_clip1.tif'
# gt = 'gt.tif'
# ai_output_1 = 'Postprocess_output.tif'
# ai_output_2 = 'AI_output.tif'

# root = 'AI20201113PHL'
# sar = 'S1A_clip1.tif'
# gt = 'gt.tif'
# ai_output_1 = 'Postprocess_output.tif'
# ai_output_2 = 'AI_output.tif'

im_sar = np.asarray(Image.open(osp.join(base, root, sar)))
im_gt = np.asarray(Image.open(osp.join(base, root, gt))).astype(int)
output_1 = cv2.cvtColor(cv2.imread(osp.join(base, root, ai_output_2)), cv2.COLOR_BGR2GRAY).astype(int)
output_2 = cv2.cvtColor(cv2.imread(osp.join(base, root, ai_output_1)), cv2.COLOR_BGR2GRAY).astype(int)

im_gt = (im_gt.max() - im_gt).astype(bool)
output_1 = (output_1.max() - output_1).astype(bool)
output_2 = (output_2.max() - output_2).astype(bool)

# intersection = np.logical_and(im_gt, im_postprocess)
# union = np.logical_or(im_gt, im_postprocess)
# iou_score = np.sum(intersection) / np.sum(union)

# water_percentage = (im_gt==1).sum() / (im_sar!=0).sum()
# water_percentage_ai = (ai_output==1).sum() / (im_sar!=0).sum()
# water_percentage_postprocess = (postprocess_output==1).sum() / (im_sar!=0).sum()

# print("Water Percentage of AI Output:{}".format(water_percentage_ai))
# print("Water Percentage of Postprocess Output:{}".format(water_percentage_postprocess))

# print("AI output:{}".format(eval_semantic_segmentation([ai_output], [im_gt])))
# print("Postprocess output:{}".format(eval_semantic_segmentation([postprocess_output], [im_gt])))

# surface_distances = compute_surface_distances(im_gt, ai_output, [1,1])
# nsd_ai = compute_surface_dice_at_tolerance(surface_distances, 1)

# surface_distances = compute_surface_distances(im_gt, postprocess_output, [1,1])
# nsd_postprocess = compute_surface_dice_at_tolerance(surface_distances, 1)
# print("NSD of ai output is {}, NSD of postprocess is {}".format(nsd_ai, nsd_postprocess))

# print("AI output confusion matrix: {}".format(confusion_matrix(im_gt.flatten(), ai_output.flatten()).ravel()))
# print("Postprocess confusion matrix: {}".format(confusion_matrix(im_gt.flatten(), postprocess_output.flatten()).ravel()))

ai_output = np.zeros((im_sar.shape[0], im_sar.shape[1], 3), dtype=int)
for i in range(im_sar.shape[0]):
    for j in range(im_sar.shape[1]):
        if im_sar[i][j] != 0:
            if output_1[i][j]:
                ai_output[i][j][0] = 7
                ai_output[i][j][1] = 47
                ai_output[i][j][2] = 108
            else:
                ai_output[i][j][0] = 165
                ai_output[i][j][1] = 165
                ai_output[i][j][2] = 165
        else:
            ai_output[i][j][0] = 0
            ai_output[i][j][1] = 0
            ai_output[i][j][2] = 0

plt.axis('off')
plt.imshow(ai_output)
plt.savefig(osp.join(base, root, 'output1.png'), bbox_inches='tight', pad_inches=0.0)
plt.clf()

postprocess_output = np.zeros((im_sar.shape[0], im_sar.shape[1], 3), dtype=int)
for i in range(im_sar.shape[0]):
    for j in range(im_sar.shape[1]):
        if im_sar[i][j] != 0:
            if output_2[i][j]:
                postprocess_output[i][j][0] = 7
                postprocess_output[i][j][1] = 47
                postprocess_output[i][j][2] = 108
            else:
                postprocess_output[i][j][0] = 165
                postprocess_output[i][j][1] = 165
                postprocess_output[i][j][2] = 165
        else:
            postprocess_output[i][j][0] = 0
            postprocess_output[i][j][1] = 0
            postprocess_output[i][j][2] = 0

plt.axis('off')
plt.imshow(postprocess_output)
plt.savefig(osp.join(base, root, 'output2.png'), bbox_inches='tight', pad_inches=0.0)
plt.clf()

gt_output = np.zeros((im_sar.shape[0], im_sar.shape[1], 3), dtype=int)
for i in range(im_sar.shape[0]):
    for j in range(im_sar.shape[1]):
        if im_sar[i][j] != 0:
            if im_gt[i][j]:
                gt_output[i][j][0] = 7
                gt_output[i][j][1] = 47
                gt_output[i][j][2] = 108
            else:
                gt_output[i][j][0] = 165
                gt_output[i][j][1] = 165
                gt_output[i][j][2] = 165
        else:
            gt_output[i][j][0] = 0
            gt_output[i][j][1] = 0
            gt_output[i][j][2] = 0

plt.axis('off')
plt.imshow(gt_output)
plt.savefig(osp.join(base, root, 'gt_output.png'), bbox_inches='tight', pad_inches=0.0)
plt.clf()

compare_map = np.zeros((im_gt.shape[0], im_gt.shape[1], 3), dtype=int)
for i in range(im_gt.shape[0]):
    for j in range(im_gt.shape[1]):
        if not output_2[i][j] and not output_1[i][j]:
            compare_map[i][j][0] = 165
            compare_map[i][j][1] = 165
            compare_map[i][j][2] = 165
        elif output_2[i][j] and not output_1[i][j]:
            compare_map[i][j][0] = 252
            compare_map[i][j][1] = 33
            compare_map[i][j][2] = 29
        elif not output_2[i][j] and output_1[i][j]:
            compare_map[i][j][0] = 0
            compare_map[i][j][1] = 255
            compare_map[i][j][2] = 1
        else:
            compare_map[i][j][0] = 7
            compare_map[i][j][1] = 47
            compare_map[i][j][2] = 108

# colormaps = ['#a5a5a5', '#072f6c', '#fd2022', '#00ff01']
# plt.imshow(compare_map)
# plt.show()
plt.axis('off')
plt.imshow(compare_map)
plt.savefig(osp.join(base, root, 'difference_map.png'), bbox_inches='tight', pad_inches=0.0)
plt.clf()