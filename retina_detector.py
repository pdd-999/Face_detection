import torch
import numpy as np
import cv2
import os

from .core import FaceDetector
from .models.retinaface import RetinaFace
from .data import cfg_mnet
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .utils.box_utils import decode, decode_landm


class RetinaDetector(FaceDetector):
    def __init__(self, device, 
                    path_to_detector="retinaface/weights/mobilenet0.25_Final.pth", 
                    mobilenet_pretrained="retinaface/weights/mobilenetV1X0.25_pretrain.tar", 
                    verbose=None):
        super().__init__(device, verbose)
        
        model = RetinaFace(cfg=cfg_mnet, phase='test', mobilenet_pretrained=mobilenet_pretrained)
        self.device = device
        
        self.origin_size = True
        self.confidence_threshold = 0.02
        self.nms_threshold = 0.4

        self.face_detector = self.load_model(model, path_to_detector)
        self.face_detector.to(device)
        self.face_detector.eval()
    
    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}
    
    def load_model(self, model, pretrained_path):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if self.device=='cpu':
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    
    def detect_from_image(self, img):
        '''
        detect face from input RGB image
        :img: ndarray
        '''
        torch.set_grad_enabled(False)
        # img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = np.float32(img)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.face_detector(img)  # forward pass
        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        for det in dets:
            det[0] = np.clip(det[0], 0, im_width)
            det[2] = np.clip(det[2], 0, im_width)
            det[1] = np.clip(det[1], 0, im_height)
            det[3] = np.clip(det[3], 0, im_height)
        
        return dets

    def detect_from_directory(self, dir):
        return

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0

if __name__=="__main__":
    detector = RetinaDetector('cpu', os.path.abspath('./weights/mobilenet0.25_Final.pth'), 
                                     os.path.abspath("./weights/mobilenetV1X0.25_pretrain.tar"))
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF
        preds = detector.detect_from_image(frame)
        try:
            for pred in preds:
                if pred[-1] >= 0.8:
                    frame = cv2.rectangle(frame, (pred[0], pred[1]), (pred[2], pred[3]), color=(0,255,0), thickness=2)
        except Exception as e:
            print(e)    

        cv2.imshow("", frame)

        if key == ord("q"):
            break