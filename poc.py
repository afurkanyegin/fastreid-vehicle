import argparse
import atexit
import bisect
import os

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.backends import cudnn

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
from fastreid.utils.logger import setup_logger

cudnn.benchmark = True
setup_logger(name="fastreid")


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """

        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()

        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # Make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5


class FeatureExtractor(object):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
            parallel (bool) whether to run the model in different processes from visualization.:
                Useful since the visualization logic can be slow.
        """
        self.cfg = cfg
        self.num_gpus = torch.cuda.device_count()
        self.predictor = AsyncPredictor(cfg, self.num_gpus)

    def run_on_image(self, original_image):
        """

        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (np.ndarray): normalized feature of the model.
        """
        original_image = original_image[:, :, ::-1]
        image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        predictions = self.predictor(image)
        return predictions


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class ObjectDetector(object):
    """
    A class for detecting objects
    """

    def __init__(self, datadir, included_objects, confidence_threshold=0.5, nms_threshold=0.3, min_object_size=100000,
                 max_object_size=float("inf")):
        """
        Creates a YOLO V3 object detector
        :param datadir: directory that includes the labels, yolo config and yolo weights
        :param included_objects: The specific labels to detect
        :param confidence_threshold: YOLO confidence threshold
        :param nms_threshold: Threshold used for NMS overlapping
        """
        label_file = os.path.join(datadir, "coco.names")
        weights_file = os.path.join(datadir, "yolov4.weights")
        cfg_file = os.path.join(datadir, "yolov4.cfg")
        self.labels = open(label_file).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
        self.included_objects = []
        for o in included_objects:
            self.included_objects.append(self.labels.index(o))
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_object_size = min_object_size
        self.max_object_size = max_object_size

    def process(self, frame):
        """
        Detects objects in a frame
        :param frame: frame to search for objects in
        :return: bounding boxes of object and a copy of the frame with bounding box drawn on it
        """
        if bad_frame(frame):
            print("object counter encountered bad frame, ignoring")
            return [], frame
        frame_height, frame_width = frame.shape[:2]

        layer_names = self.net.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (224, 224), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(layer_names)

        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id not in self.included_objects:
                    continue
                if confidence > self.confidence_threshold:
                    box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (x, y, w, h) = box.astype("int")
                    if w * h < self.min_object_size or w * h > self.max_object_size:
                        continue
                    x = int(x - (w / 2))
                    y = int(y - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        return_boxes = []
        validation_image = frame.copy()
        if len(idxs) > 0:
            for i in idxs.flatten():
                return_boxes.append(boxes[i])
        draw_boxes(validation_image, return_boxes)
        return return_boxes, validation_image


def draw_boxes(frame, boxes, color=(0, 255, 0)):
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + int(w / 2), y + int(h / 2)), 4, color, -1)


def bad_frame(frame):
    """
    Detect if a frame is bad. In this case, bad is a frame that is mostly gray (128, 128, 128)
    or mostly black
    :param frame: the frame to check if it's bad
    :return: True if frame is bad, false otherwise
    """
    b, g, r = cv2.split(frame)
    # gray frame
    if np.std(r) < 20 and np.std(g) < 20 and np.std(b) < 20 and np.median(r) == np.median(g) == np.median(b) == 128:
        return True
    # black frame
    if np.std(r) < 10 and np.std(g) < 10 and np.std(b) < 10 and np.median(r) == np.median(g) == np.median(b) < 25:
        return True
    return False


class Vehicle(object):
    def __init__(self, feature, image, id, last_seen):
        self.features = [feature]
        self.images = [image]
        self.id = id
        self.last_seen = last_seen

    def compare_features(self, feature):
        results = []
        for feat in self.features:
            distmat = 1 - torch.mm(feat, feature.t())
            score = 1 - distmat[0][0]
            results.append(score)
        return np.average(results)

    def add(self, feature, image):
        self.features.append(feature)
        self.images.append(image)

    def merge(self, vehicle: "Vehicle"):
        for i in range(len(vehicle.features)):
            feat = vehicle.features[i]
            img = vehicle.images[i]
            self.add(feat, img)

    def compare_vehicles(self, vehicle: "Vehicle"):
        score = 0
        for feat in vehicle.features:
            feat_score = self.compare_features(feat)
            score += feat_score
        return score / float(len(vehicle.features))


def main(interval):
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractor(cfg)
    detector = ObjectDetector("/Users/ianhoegen/presto_configs/input/yolo", ["car", "truck", "motorbike"],
                              min_object_size=0)

    vehicles = []
    assert len(args.input) > 0

    # for vid in args.input[0]
    sources = []
    fps = 0
    for vid in args.input:
        capture = cv2.VideoCapture(vid)
        sources.append(capture)
        fps = int(capture.get(cv2.CAP_PROP_FPS))

    fps /= float(interval)
    print("running at " + str(fps) + "fps")
    outs = []

    count = -1
    run = True
    last_created_updated_time = 0
    last_created_vehicle = None
    time_last_seen = {}
    removed_vehicles = set()
    while run:
        frames = []
        for i, cap in enumerate(sources):
            ret, cap_frame = cap.read()
            frames.append(cap_frame)
            if len(outs) < len(sources):
                height, width, _ = cap_frame.shape
                out = cv2.VideoWriter("vid_veri" + str(fps) + "fps_" + str(i) + ".mp4",
                                      cv2.VideoWriter_fourcc(*'mp4v'), fps * 3, (width, height))
                outs.append(out)
            if not ret or cap_frame is None:
                run = False
                break
        if not run:
            break
        count += 1
        if count % interval != 0:
            continue
        combined_val = None
        for p, frame in enumerate(frames):
            boxes, val = detector.process(frame)
            if len(boxes) > 0:
                for box in boxes:
                    x, y, w, h = box
                    cropped_image = frame[max(y, 0):y + h, max(x, 0):x + w]
                    feat = demo.run_on_image(cropped_image)
                    best_id = None
                    max_score = 1
                    if len(vehicles) == 0:
                        if p == 0:
                            best_id = len(vehicles)
                            best = Vehicle(feat, cropped_image, len(vehicles), count)
                            last_created_vehicle = best
                            last_created_updated_time = count
                            vehicles.append(best)
                    else:
                        best = None
                        max_score = float('-inf')
                        for vehicle in vehicles:
                            if vehicle.id in removed_vehicles:
                                continue
                            score = vehicle.compare_features(feat)
                            if score > max_score:
                                max_score = score
                                best = vehicle
                        if (max_score >= 0.5) and count - best.last_seen > (interval - 1):
                            best.add(feat, cropped_image)
                            best.last_seen = count
                            best_id = best.id
                            if p == 1:
                                time_last_seen[best_id] = count
                            elif p == 0:
                                last_created_updated_time = count
                        elif p == 0:
                            if count - last_created_updated_time > 15:
                                best_id = len(vehicles)
                                print("new vehicle", best_id)
                                max_score = 1 - max_score
                                new_vehicle = Vehicle(feat, cropped_image, len(vehicles), count)
                                last_created_vehicle = new_vehicle
                                last_created_updated_time = count
                                vehicles.append(new_vehicle)
                            else:
                                best_id = last_created_vehicle.id
                                print("recently created vehicle", best_id, "adding to that instead")
                                last_created_updated_time = count
                                last_created_vehicle.add(feat, cropped_image)
                    if best_id is not None:
                        cv2.putText(val, "ID: " + str(best_id) + " confidence: " + str(max_score * 100)[:5] + "%",
                                    (x, y + h), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3, cv2.LINE_AA)
            outs[p].write(val)
        # cv2.imshow("video", combined_val)
        if count % 150 == 0:
            to_merge = []
            for i in range(len(vehicles) - 1):
                vehicle = vehicles[i]
                for j in range(i + 1, len(vehicles)):
                    vehicle_2 = vehicles[j]
                    comp_score = vehicle.compare_vehicles(vehicle_2)
                    if comp_score > 0.5:
                        to_merge.append((i, j))
            for i, j in to_merge:
                if i in removed_vehicles or j in removed_vehicles:
                    continue
                print("merging", i, j)
                vehicles[i].merge(vehicles[j])
                removed_vehicles.add(j)
        for v in time_last_seen:
            t = time_last_seen[v]
            if count - t >= 900 and v not in removed_vehicles:
                print("removing", v, "because haven't seen in", (count - t) / 15.0, "seconds")
                removed_vehicles.add(v)
        # cv2.waitKey(1)
    for cap in sources:
        cap.release()
    for out in outs:
        out.release()


# intervals = [5]
# intervals = [15, 5]
intervals = [15]
import time
for inv in intervals:
    start = time.time()
    main(inv)
    print((time.time() - start)/(60*60))
