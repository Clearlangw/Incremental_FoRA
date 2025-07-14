import ultralytics.global_mode as gb
import sys
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import torch
from ultralytics import YOLO_m, YOLO
import numpy as np
from ultralytics.data.augment import LetterBox
import os
from ultralytics.data import load_inference_source
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
import cv2

gb.set_global_mode('rgb')
device = 'cpu'  # 'cuda:0'
imgsz = 800
augment = False
# model = YOLO('yolov8n.pt')
# model.predict('M1401_001050.jpg', device='cpu')

model = YOLO_m('/root/multi_head_yolov8n_lma_weights/weights/best.pt')
# print(model)
# img_paths = ['M1401_001050.jpg', 'M1401_001050.jpg']  # 单模态单张或list，双模态list
img_paths = 'M1401_001050.bmp'


# # 使用绝对路径
# model.predict('M1401_001050.jpg', device='cpu')


# # mode = gb.global_mode


def predict_model(model, img_paths, device, mode='fuse', prefer='rgb', augment=False, imgsz=800):
    #
    # self.device = self.model.device  # update device
    # self.args.half = self.model.fp16  # update half
    # print(model)
    model.model = model.model.to(device)
    model.model.modal = mode
    if mode == 'fuse':
        batch_rgb = next(iter(load_inference_source(img_paths[0])))
        batch_ir = next(iter(load_inference_source(img_paths[1])))
        path_rgb, rgb_im0s, vid_cap_rgb, s_rgb = batch_rgb
        path_ir, ir_im0s, vid_cap_ir, s_ir = batch_ir
        # print(path_rgb)
        # sys.exit()
        im_rgb = preprocess(model, rgb_im0s, device, imgsz)  # TODO:这里要改
        im_ir = preprocess(model, ir_im0s, device, imgsz)
        preds = inference(model, im_rgb, im_ir, augment=augment)
        if prefer == 'rgb':
            results = postprocess(model, preds, im_rgb, rgb_im0s, img_paths[0])  # 默认使用rgb
            im = im_rgb
        else:
            results = postprocess(model, preds, im_ir, ir_im0s, img_paths[1])  # 默认使用rgb
            im = im_ir
    else:
        my_pth = None
        if isinstance(img_paths, list):
            if mode == 'rgb':
                batch = next(iter(load_inference_source(img_paths[0])))
                my_pth = img_paths[0]
            else:
                batch = next(iter(load_inference_source(img_paths[1])))
                my_pth = img_paths[1]
        else:
            batch = next(iter(load_inference_source(img_paths)))
            my_pth = img_paths
        path, im0s, vid_cap, s = batch
        im = preprocess(model, im0s, device, imgsz)
        preds = inference(model, im, im, augment=augment)
        results = postprocess(model, preds, im, im0s, my_pth)
    return results, im


def pre_transform(model, im, imgsz):
    """
    Pre-transform input image before inference.

    Args:
        im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (list): A list of transformed images.
    """
    same_shapes = all(x.shape == im[0].shape for x in im)
    letterbox = LetterBox(imgsz, auto=same_shapes, stride=32)
    return [letterbox(image=x) for x in im]


def preprocess(model, im, device, imgsz):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(model, im, imgsz))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
    half = False
    im = im.to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def inference(model, im0, im1, augment=False):
    # print(model.model.modal)
    return model.model.predict(im0, im1, augment=augment)


def postprocess(model, preds, img, orig_imgs, img_path, conf=0.15, iou=0.45):
    """Post-processes predictions and returns a list of Results objects."""
    preds = ops.non_max_suppression(
        preds,
        conf,  # self.conf
        iou,  # self.args.iou
        agnostic=False,
        max_det=300,
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        img_path = os.path.abspath(img_path)
        results.append(Results(orig_img, path=img_path, names=model.names, boxes=pred))
    return results


def vis_results(results, im, save_dir, img_paths, mode='fuse', prefer='rgb'):
    result = results[0]
    print(img_paths)
    if mode == 'fuse':
        if prefer == 'rgb':
            file_name = os.path.basename(img_paths[0])
        else:
            file_name = os.path.basename(img_paths[1])
    else:
        if isinstance(img_paths, list):
            if mode == 'rgb':
                file_name = os.path.basename(img_paths[0])
            else:
                file_name = os.path.basename(img_paths[1])
        else:
            file_name = os.path.basename(img_paths)

    plot_args = {
        "line_width": None,
        "boxes": True,
        "conf": True,
        "labels": True,
    }

    plot_args["im_gpu"] = im[0]
    plotted_img = result.plot(**plot_args)
    print(file_name)
    # print(plotted_img)
    save_path = os.path.join(save_dir, file_name)
    print(save_path)
    cv2.imwrite(save_path, plotted_img)
    return save_path,plotted_img


results, im = predict_model(model, img_paths, device, mode=gb.read_global_mode())
# print(len(results))
# print(results[0].boxes)
savepath,pltimg = vis_results(results, im, "..", img_paths, mode=gb.read_global_mode())
