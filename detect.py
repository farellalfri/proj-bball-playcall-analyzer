from ultralytics import YOLO

# Load the "nano" model (the smallest and fastest version)
# It downloads automatically the first time you run this.
model = YOLO('yolov8n.pt')

# Run the model on your basketball video
# source: path to your video file
# save: saves the output video with boxes drawn
# classes: [0, 32] tells it to only look for 'person' (0) and 'sports ball' (32)
results = model.predict(source='basketball_game2.mp4', save=True)