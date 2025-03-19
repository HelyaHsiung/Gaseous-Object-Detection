from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np

from datasets.init_dataset import get_dataset

from opts import opts
from ACT_utils.ACT_utils import iou2d, pr_to_ap
from ACT_utils.ACT_build import load_frame_detections, BuildTubes

def frameAP(opt, print_info=True):
    redo = opt.redo
    th = opt.th
    split = 'val'
    model_name = opt.model_name
    Dataset = get_dataset(opt.dataset)
    dataset = Dataset(opt, split)

    inference_dirname = opt.inference_dir
    print('inference_dirname is ', inference_dirname)
    print('threshold is ', th)

    vlist = dataset._test_videos[opt.split - 1]

    # load per-frame detections
    frame_detections_file = os.path.join(inference_dirname, 'frame_detections.pkl')
    if os.path.isfile(frame_detections_file) and not redo:
        print('load previous linking results...')
        print('if you want to reproduce it, please add --redo')
        with open(frame_detections_file, 'rb') as fid:
            alldets = pickle.load(fid)
    else:
        alldets = load_frame_detections(opt, dataset, opt.K, vlist, inference_dirname)
        try:
            with open(frame_detections_file, 'wb') as fid:
                pickle.dump(alldets, fid, protocol=4)
        except:
            print("OverflowError: cannot serialize a bytes object larger than 4 GiB")

    results = {}
    # compute AP for each class
    for ilabel, label in enumerate(dataset.labels):
        # detections of this class
        detections = alldets[alldets[:, 2] == ilabel, :]

        # load ground-truth of this class
        gt = {}
        for iv, v in enumerate(vlist):
            tubes = dataset._gttubes[v]

            if ilabel not in tubes:
                continue

            for tube in tubes[ilabel]:
                for i in range(tube.shape[0]):
                    k = (iv, int(tube[i, 0]))
                    if k not in gt:
                        gt[k] = []
                    gt[k].append(tube[i, 1:5].tolist())

        for k in gt:
            gt[k] = np.array(gt[k])
        gt_s = gt.copy()
        gt_m = gt.copy()
        gt_l = gt.copy()

        # pr will be an array containing precision-recall values
        pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
        pr[0, 0] = 1.0
        pr[0, 1] = 0.0
        fn = sum([g.shape[0] for g in gt.values()])  # false negatives
        fp = 0  # false positives
        tp = 0  # true positives

        # small: area < 32 * 32; middle: 32 * 32 < area < 96 * 96; large: area > 96 * 96;
        pr_s = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
        pr_s[0, 0] = 1.0
        pr_s[0, 1] = 0.0
        pr_m = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
        pr_m[0, 0] = 1.0
        pr_m[0, 1] = 0.0
        pr_l = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
        pr_l[0, 0] = 1.0
        pr_l[0, 1] = 0.0
        fn_s, fp_s, tp_s, fn_m, fp_m, tp_m, fn_l, fp_l, tp_l = 0, 0, 0, 0, 0, 0, 0, 0, 0

        for k in gt.keys():
            area = (gt[k][0][3] - gt[k][0][1] + 1) * (gt[k][0][2] - gt[k][0][0] + 1)
            if area < 32*32:
                fn_s += 1
                del gt_m[k]
                del gt_l[k]
            elif area < 96*96:
                fn_m += 1
                del gt_s[k]
                del gt_l[k]
            else:
                fn_l += 1
                del gt_s[k]
                del gt_m[k]

        gt_s_ori = gt_s.copy()
        gt_m_ori = gt_m.copy()
        gt_l_ori = gt_l.copy()
        count_s, count_m, count_l = 0, 0, 0

        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]

            ispositive = False
            if k in gt:
                ious = iou2d(gt[k], box)
                amax = np.argmax(ious)
                if ious[amax] >= th:
                    ispositive = True
                    gt[k] = np.delete(gt[k], amax, 0)
                    if gt[k].size == 0:
                        del gt[k]

            if ispositive:
                tp += 1
                fn -= 1
            else:
                fp += 1
            pr[i + 1, 0] = float(tp) / float(tp + fp)
            pr[i + 1, 1] = float(tp) / float(tp + fn)
        results[label] = pr

        #small
        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]

            if k not in gt_s_ori:
                continue
            ispositive = False
            if k in gt_s:
                ious = iou2d(gt_s[k], box)
                amax = np.argmax(ious)
                if ious[amax] >= th:
                    ispositive = True
                    gt_s[k] = np.delete(gt_s[k], amax, 0)
                    if gt_s[k].size == 0:
                        del gt_s[k]
            if ispositive:
                tp_s += 1
                fn_s -= 1
            else:
                fp_s += 1
            pr_s[count_s + 1, 0] = float(tp_s) / float(tp_s + fp_s)
            pr_s[count_s + 1, 1] = float(tp_s) / float(tp_s + fn_s)
            count_s += 1

        #middle
        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]

            if k not in gt_m_ori:
                continue
            ispositive = False
            if k in gt_m:
                ious = iou2d(gt_m[k], box)
                amax = np.argmax(ious)

                if ious[amax] >= th:
                    ispositive = True
                    gt_m[k] = np.delete(gt_m[k], amax, 0)
                    if gt_m[k].size == 0:
                        del gt_m[k]

            if ispositive:
                tp_m += 1
                fn_m -= 1
            else:
                fp_m += 1
            pr_m[count_m + 1, 0] = float(tp_m) / float(tp_m + fp_m)
            pr_m[count_m + 1, 1] = float(tp_m) / float(tp_m + fn_m)
            count_m += 1

        #large
        for i, j in enumerate(np.argsort(-detections[:, 3])):
            k = (int(detections[j, 0]), int(detections[j, 1]))
            box = detections[j, 4:8]
            if k not in gt_l_ori:
                continue
            ispositive = False
            if k in gt_l:
                ious = iou2d(gt_l[k], box)
                amax = np.argmax(ious)
                if ious[amax] >= th:
                    ispositive = True
                    gt_l[k] = np.delete(gt_l[k], amax, 0)
                    if gt_l[k].size == 0:
                        del gt_l[k]

            if ispositive:
                tp_l += 1
                fn_l -= 1
            else:
                fp_l += 1
            pr_l[count_l + 1, 0] = float(tp_l) / float(tp_l + fp_l)
            pr_l[count_l + 1, 1] = float(tp_l) / float(tp_l + fn_l)
            count_l += 1

    # display results
    ap = 100 * np.array([pr_to_ap(results[label]) for label in dataset.labels])
    ap_s = 100 * np.array([pr_to_ap(pr_s)])
    ap_m = 100 * np.array([pr_to_ap(pr_m)])
    ap_l = 100 * np.array([pr_to_ap(pr_l)])
    frameap_result = np.mean(ap)
    frameap_result_s = np.mean(ap_s)
    frameap_result_m = np.mean(ap_m)
    frameap_result_l = np.mean(ap_l)
    if print_info:
        log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
        log_file.write('\nTask_{} frameAP_{}\n'.format(model_name, th))
        print('Task_{} frameAP_{}\n'.format(model_name, th))
        log_file.write("\n{:20s} {:8.2f}\n\n".format("mAP", frameap_result))
        log_file.close()
        print("{:20s} {:8.2f}".format("mAP", frameap_result))
        print("{:20s} {:8.2f}".format("area=small mAP", frameap_result_s))
        print("{:20s} {:8.2f}".format("area=middle mAP", frameap_result_m))
        print("{:20s} {:8.2f}".format("area=large mAP", frameap_result_l))

    return [frameap_result,frameap_result_s,frameap_result_m,frameap_result_l]

def frameAP_050_095(opt):
    ap, ap_s, ap_m, ap_l = 0, 0, 0, 0
    for i in range(10):
        opt.th = 0.5 + 0.05 * i
        frameAP_allresult = frameAP(opt, print_info=False)
        ap += frameAP_allresult[0]
        ap_s += frameAP_allresult[1]
        ap_m += frameAP_allresult[2]
        ap_l += frameAP_allresult[3]
    ap = ap / 10.0
    ap_s = ap_s / 10.0
    ap_m = ap_m / 10.0
    ap_l = ap_l / 10.0
    log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
    log_file.write('\nTask_{} FrameAP_0.50:0.95 \n'.format(opt.model_name))
    log_file.write("\n{:20s} {:8.2f}\n\n".format("mAP", ap))
    log_file.close()
    print('Task_{} FrameAP_0.50:0.95 \n'.format(opt.model_name))
    print("\n{:20s} {:8.2f}\n\n".format("mAP", ap))
    print("\n{:20s} {:8.2f}\n\n".format("area=small mAP", ap_s))
    print("\n{:20s} {:8.2f}\n\n".format("area=middle mAP", ap_m))
    print("\n{:20s} {:8.2f}\n\n".format("area=large mAP", ap_l))

if __name__ == "__main__":
    opt = opts().parse()
    if not os.path.exists(os.path.join(opt.root_dir, 'result')):
        os.system("mkdir -p '" + os.path.join(opt.root_dir, 'result') + "'")
    if opt.task == 'BuildTubes':
        BuildTubes(opt)
    elif opt.task == 'frameAP':
        frameAP(opt)
    elif opt.task == 'frameAP_all':
        frameAP_050_095(opt)
    else:
        raise NotImplementedError('Not implemented:' + opt.task)
