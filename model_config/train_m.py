import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


model = YOLO_m("/root/multi_head_yolov8n_lma_weights/weights/last.pt")
# model = YOLO_m("/root/FoRA/yolov8n_lma_multi_head.yaml")
# print(model.model.modal)
model.model.modal = 'both'

model.train(data="drone_vehicle_m.yaml",
            epochs=50,
            patience=30,
            batch=80,
            imgsz=800,
            device=[0,1],
            r_init=9,
            r_target=6,
            adalora=True,
            project="DroneVehicle",
            name='only_test',
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            resume=True,
            )

