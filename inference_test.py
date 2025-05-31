from ultralytics import YOLO

model = YOLO('seatscout_v5.pt')

results = model('Room Videos/room1.mp4', show=True, save=True, project='v5_inference', name='predict')