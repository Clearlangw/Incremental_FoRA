from ultralytics import YOLO

model = YOLO("yolov8x.yaml")
# print(model)
model.train(data="drone_vehicle.yaml",
            epochs=50,
            patience=30,
            batch=56,
            device=[0, 1],
            imgsz=640,
            # r_init=24,
            # r_target=6,
            # adalora=False,
            exist_ok=True,
            project="DroneVehicle",
            name='TIF_yolov8x',
            resume=False,
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            )
