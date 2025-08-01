import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt")
model.model.modal = 'both'
model.train(data="/root/FoRA/data_yaml/fewshot_dv_vedai.yaml",
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="incremental_old_model_work_dir",
            name='allvedai_tfa',
            optimizer='SGD',
            seed=0,
            freeze=55,
            )

model = YOLO_m("/root/FoRA/incremental_exp/incremental_old_model_work_dir/allvedai_tfa/weights/best.pt")
model.model.modal = 'both'
model.val(data="/root/FoRA/data_yaml/incremental_vedai2dvvedai.yaml",
          project="incremental_old_model_work_dir",
          name='allvedai_tfa_test',
          imgsz=800,
          batch=20,
          device=[0]
          )