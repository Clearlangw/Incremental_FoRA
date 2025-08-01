import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m
print("@"*100)
print("开始训练,第一组,均为vedai的500张图像,对应红表")
print("@"*100)
print("@#$")
model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt",
            incremental_yaml="/root/FoRA/model_yaml/incremental_yolov8h_lma_multi_head_ScaleConvFuse.yaml")
model.model.modal = 'both'
model.train(data="/root/FoRA/data_yaml/fewshot_dv_vedai.yaml",
            use_contrastive=True,
            use_prototype=True,
            use_negative_weighted_contrastive=True,
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="incremental_old_model_work_dir",
            name='fewshot_allvedai_tfa_gfsd_contrast_prototype',
            optimizer='SGD',
            #lr0 = 0.005,#reload的话当时设置的是0.005,这部分可以注释
            seed=0,
            freeze=55,
            )

##测试
model = YOLO_m("/root/FoRA/incremental_exp/incremental_old_model_work_dir/fewshot_allvedai_tfa_gfsd_contrast_prototype/weights/best.pt")
model.val(data="/root/FoRA/data_yaml/incremental_vedai2dvvedai.yaml",
          project="incremental_old_model_work_dir",
          name='fewshot_allvedai_tfa_gfsd_contrast_prototype_test',
          imgsz=800,
          batch=20,
          device=[0]
          )

print("@"*100)
print("开始训练,第二组,250shots,对应黑表")
print("@"*100)
print("@#$")
model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt",
            incremental_yaml="/root/FoRA/model_yaml/incremental_yolov8h_lma_multi_head_ScaleConvFuse.yaml")
model.model.modal = 'both'
model.train(data="/root/FoRA/data_yaml/250shots_incremental_vedai.yaml",
            use_contrastive=True,
            use_prototype=True,
            use_negative_weighted_contrastive=True,
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="incremental_old_model_work_dir",
            name='250shots_tfa_gfsd_contrast_prototype',
            optimizer='SGD',
            #lr0 = 0.005,#reload的话当时设置的是0.005,这部分可以注释
            seed=0,
            freeze=55,
            )

##测试
model = YOLO_m("/root/FoRA/incremental_exp/incremental_old_model_work_dir/250shots_tfa_gfsd_contrast_prototype/weights/best.pt")
model.val(data="/root/FoRA/data_yaml/incremental_vedai2dvvedai.yaml",
          project="incremental_old_model_work_dir",
          name='250shots_tfa_gfsd_contrast_prototype_test',
          imgsz=800,
          batch=20,
          device=[0]
          )

print("@"*100)
print("开始训练,第三组,文浩shots,对应蓝表")
print("@"*100)
print("@#$")
model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt")
model.model.modal = 'both'
model.train(data="/root/FoRA/data_yaml/wenhao_incremental_vedai.yaml",
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="incremental_old_model_work_dir",
            name='wenhao_tfa',
            optimizer='SGD',
            seed=0,
            freeze=55,
            )

model = YOLO_m("/root/FoRA/incremental_exp/incremental_old_model_work_dir/wenhao_tfa/weights/best.pt")
model.model.modal = 'both'
model.val(data="/root/FoRA/data_yaml/incremental_vedai2dvvedai.yaml",
          project="incremental_old_model_work_dir",
          name='wenhao_tfa_test',
          imgsz=800,
          batch=20,
          device=[0]
          )