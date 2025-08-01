import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


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