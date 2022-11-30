import glob

dir_path = 'data/'  # Your images should be here!
img = f"{dir_path}*.jpg"
weights = glob.glob("yolo/*.weights")[0]
labels = glob.glob("yolo/*.txt")[0]
cfg = glob.glob("yolo/*.cfg")[0]

CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.4
