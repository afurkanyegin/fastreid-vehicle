import cv2


def combine_cameras(cam_num):
    files = [
        "/Users/ianhoegen/presto_configs/input/bench/10-30-2020_1200-1300/video/cam" + cam_num + "/2020-10-30_12.00.mkv",
        "/Users/ianhoegen/presto_configs/input/bench/10-30-2020_1200-1300/video/cam" + cam_num + "/2020-10-30_12.15.mkv",
        "/Users/ianhoegen/presto_configs/input/bench/10-30-2020_1200-1300/video/cam" + cam_num + "/2020-10-30_12.30.mkv",
        "/Users/ianhoegen/presto_configs/input/bench/10-30-2020_1200-1300/video/cam" + cam_num + "/2020-10-30_12.45.mkv",
    ]

    caps = []
    out = None
    fps = 0

    for file in files:
        capture = cv2.VideoCapture(file)
        caps.append(capture)
        fps = int(capture.get(cv2.CAP_PROP_FPS))

    run = True
    cap_idx = 0

    while run:
        cap = caps[cap_idx]
        ret, frame = cap.read()
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter("10-30-2020_1200-1300-CAM" + cam_num + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                  (width, height))
        if not ret or frame is None:
            print("doing next capture")
            cap_idx += 1
            if cap_idx >= len(caps):
                run = False
                break
        if not run:
            break
        out.write(frame)

    for cap in caps:
        cap.release()
    out.release()


cams = ["03", "04"]
for cam in cams:
    print("combining for", cam)
    combine_cameras(cam)
