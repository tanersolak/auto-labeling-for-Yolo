import glob

dir_path = 'data/'        # Path for images
img = f"{dir_path}*.jpg"  # It can be changed as your images type

weights = glob.glob("yolo/*.weights")[0]  # Path for weights
labels = glob.glob("yolo/*.txt")[0]       # Path for labels
cfg = glob.glob("yolo/*.cfg")[0]          # Path for cfg

CONFIDENCE_THRESHOLD = 0.30
NMS_THRESHOLD = 0.4
