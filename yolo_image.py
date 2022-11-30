import glob
import os
import cv2
import numpy as np
import SETTINGS

CONFIDENCE_THRESHOLD = SETTINGS.CONFIDENCE_THRESHOLD
NMS_THRESHOLD = SETTINGS.NMS_THRESHOLD

dir_path = SETTINGS.dir_path
img = SETTINGS.img
weights = SETTINGS.weights
labels = SETTINGS.labels
cfg = SETTINGS.cfg

print("You are now using {} weights ,{} configs and {} labels.".format(weights, cfg, labels))

lbls = list()
with open(labels, "r") as f:
    lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# count = len(fnmatch.filter(os.listdir(dir_path), '*.jpg'))


def detect(nn, imgpath):
    image = cv2.imread(imgpath)
    # image = cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA)

    assert image is not None, f"Image is none, check file path. Given path is: {imgpath}"
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    nn.setInput(blob)
    layer_outs = nn.forward(layer)

    boxes = list()
    confidences = list()
    class_ids = list()

    for output in layer_outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    image_name = os.path.splitext(imgpath)
    if len(idxs) > 0:
        file = open(f"{image_name[0]}.txt", "w+")

        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dh = 1./image.shape[0]
            dw = 1./image.shape[1]
            norm_x = round((x + int(w / 2)) * dw, 6)
            norm_y = round((y + int(h / 2)) * dh, 6)
            norm_dw = round(w*dw, 6)
            norm_dh = round(h*dh, 6)

            file.write(f"{class_ids[i]} {norm_x} {norm_y} {norm_dw} {norm_dh}\n")

        file.close()
    elif len(idxs) == 0:
       os.remove(imgpath)


if __name__ == "__main__":

    # print('File Count:', count)
    for filename in glob.glob(img):
        detect(net, filename)
    print("Done...")
