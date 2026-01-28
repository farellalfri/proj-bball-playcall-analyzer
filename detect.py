from ultralytics import YOLO

# Load the "nano" model (the smallest and fastest version)
# It downloads automatically the first time you run this.
model = YOLO("models/ball_detector_model.pt")

# Run the model on your basketball video
# source: path to your video file
# save: saves the output video with boxes drawn
results = model.track(source='basketball_game2.mp4', save=True)
print(results)
print("====================")
for box in results[0].boxes:
  print(box)
