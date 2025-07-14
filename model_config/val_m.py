import ultralytics.global_mode as gb
gb.set_global_mode('rgb')
from ultralytics import YOLO_m

model = YOLO_m('/root/multi_head_yolov8n_lma_weights/weights/best.pt')
# print(model.model.modal)
model.model.modal = 'rgb'
# print(model.model.modal)

data = 'drone_vehicle_m.yaml'
model.val(data=data,
          project='DroneVehicle',
          name='val_test_rgb',
          imgsz=800,
          batch=40,
          device=[0]
          # visualize=True,
          # split='val_rgb',
          )
