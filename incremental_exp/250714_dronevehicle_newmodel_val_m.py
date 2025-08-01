import ultralytics.global_mode as gb
gb.set_global_mode('both')
from ultralytics import YOLO_m

model = YOLO_m('/root/FoRA/car_and_plane_new_model/new_model_train/weights/best.pt')#因为另一个模型实际上没有训完
# print(model.model.modal)
model.model.modal = 'both'
# print(model.model.modal)

data = 'drone_vehicle_m.yaml'
model.val(data=data,
          project="test_newmodel_on_drone_vehicle",
          name='new_model_val_both',
          imgsz=800,
          batch=2,
          device=[0]
          # visualize=True,
          # split='val_both',
          )
