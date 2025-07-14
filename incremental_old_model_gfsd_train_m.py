import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


#model = YOLO_m("/root/FoRA/yolov8n_lma_multi_head.yaml")
#model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt",incremental_yaml="incremental_yolov8h_lma_multi_head_ScaleConvFuse.yaml")
model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt",incremental_yaml="incremental_yolov8h_lma_multi_head_ScaleConvFuse.yaml")
# print(model.model.modal)
model.model.modal = 'both'

model.train(data="fewshot_dv_vedai.yaml",
            incremental_yaml = "incremental_yolov8h_lma_multi_head_ScaleConvFuse.yaml",#增量的模型
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="incremental_old_model_gfsd_train_m",
            name='not_pretrained_on_coco',
            pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            # resume=True,
            )
