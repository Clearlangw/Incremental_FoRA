from ultralytics import YOLO

model = YOLO("yolov8h.yaml")
# print(model)
model.train(data="drone_vehicle.yaml",
            epochs=50,
            patience=30,
            batch=8,
            imgsz=640,
            device=0,
            # r_init=24,
            # r_target=6,
            # adalora=False,
            project="DroneVehicle",
            name='yolov8l_ir_e50_bs8',
            resume=False,
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            )