import ultralytics.global_mode as gb
gb.set_global_mode('both')
from ultralytics import YOLO_m

model = YOLO_m('/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt')#因为另一个模型实际上没有训完
# print(model.model.modal)
model.model.modal = 'both'
# print(model.model.modal)

data = 'drone_vehicle_m.yaml'
model.val(data=data,
          project="test_oldmodel_on_drone_vehicle",
          name='notpretrained_old_model_val_both',
          imgsz=800,
          batch=2,
          device=[0]
          # visualize=True,
          # split='val_both',
          )

