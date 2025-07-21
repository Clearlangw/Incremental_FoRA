import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


#model = YOLO_m("/root/FoRA/yolov8n_lma_multi_head.yaml")

model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt")
#model = YOLO_m("yolov8h_incremental_lma_multi_head_ScaleConvFuse.yaml")
# print(model.model.modal)
model.model.modal = 'both'

model.train(data="vedaifewshot_dv_vedai.yaml",
            #incremental_yaml = "yolov8h_incremental_lma_multi_head_ScaleConvFuse.yaml",#增量的模型
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="incremental_old_model_gfsd_train_m",
            name='test_old_model',
            # pretrained=False,
            optimizer='SGD',
            # lr0 = 0.2,
            seed=0,
            freeze=19,
            # resume=True,
            )
