import ultralytics.global_mode as gb
gb.set_global_mode('rgb')
from ultralytics import YOLO_m

model = YOLO_m('/root/FoRA/car_and_plane_new_model/new_model_train/weights/best.pt')
# print(model.model.modal)
model.model.modal = 'rgb'
# print(model.model.modal)

data = 'car_and_plane_m.yaml'
model.val(data=data,
          project="car_and_plane_new_model",
          name='new_model_val_rgb',
          imgsz=800,
          batch=2,
          device=[0,1]
          # visualize=True,
          # split='val_rgb',
          )
