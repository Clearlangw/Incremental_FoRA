import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


model = YOLO_m("/root/FoRA/250611newmodel_multi_head_ScaleConvFuse.yaml")
# print(model.model.modal)
model.model.modal = 'both'

model.train(data='car_and_plane_m.yaml',
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="car_and_plane_new_model",
            name='new_model_train_only_bug_test',
            exist_ok=True,
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            resume=True,
            )

# model.train(data="drone_vehicle_m.yaml",
#             epochs=50,
#             patience=30,
#             batch=4,
#             imgsz=800,
#             device=[0,1],
#             r_init=9,
#             r_target=6,
#             adalora=True,
#             project="DroneVehicle_new_model",
#             name='new_model_train',
#             exist_ok=True,
#             pretrained=False,
#             optimizer='auto',
#             seed=0,
#             freeze=None,
#             resume=True,
#             )

