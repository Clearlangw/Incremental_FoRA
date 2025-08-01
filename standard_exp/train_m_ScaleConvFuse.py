import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


model = YOLO_m("/root/FoRA/yolov8h_lma_multi_head_ScaleConvFuse.yaml")
# print(model.model.modal)
model.model.modal = 'both'

model.train(data="drone_vehicle_m.yaml",
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="DroneVehicle",
            name='ScaleConvFuse',
            exist_ok=True,
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            resume=True,
            )

