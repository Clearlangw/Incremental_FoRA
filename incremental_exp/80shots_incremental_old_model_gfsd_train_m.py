import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


#model = YOLO_m("/root/FoRA/yolov8n_lma_multi_head.yaml")
#model = YOLO_m("yolov8h_incremental_lma_multi_head_ScaleConvFuse.yaml")
#print(model.model.modal)

model = YOLO_m("/root/autodl-tmp/Original_DroneVehicle_workdir/ScaleConvFuse/weights/best.pt",incremental_yaml="incremental_yolov8h_lma_multi_head_ScaleConvFuse.yaml")
model.model.modal = 'both'
model.train(data="80shots_incremental_vedai.yaml",
            #incremental_yaml = "yolov8h_incremental_lma_multi_head_ScaleConvFuse.yaml",#增量的模型
            epochs=80,
            patience=10,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="incremental_old_model_gfsd_train_m",
            name='not_only_test_bug_80shots_normal_incremental_vedai2dvvedai',
            # pretrained=False,
            optimizer='SGD',
            lr0 = 0.005,
            seed=0,
            freeze=55,
            # resume=True,
            )
##恢复
# model = YOLO_m("/root/FoRA/incremental_old_model_gfsd_train_m/80shots_normal_incremental_vedai2dvvedai/weights/best.pt")
# model.model.modal = 'both'
# model.train(data="80shots_incremental_vedai.yaml",
#             #incremental_yaml = "yolov8h_incremental_lma_multi_head_ScaleConvFuse.yaml",#增量的模型
#             epochs=200,
#             patience=10,
#             batch=2,
#             imgsz=800,
#             device=[0],
#             r_init=9,
#             r_target=6,
#             adalora=True,
#             project="incremental_old_model_gfsd_train_m",
#             name='normal_incremental_vedai2dvvedai',
#             # pretrained=False,
#             optimizer='SGD',
#             lr0 = 0.05,
#             seed=0,
#             freeze=55,
#             resume=True,
#             )
#测试
# model = YOLO_m("/root/FoRA/incremental_old_model_gfsd_train_m/80shots_normal_incremental_vedai2dvvedai/weights/best.pt")
# model.model.modal = 'both'
# model.val(data="incremental_vedai2dvvedai.yaml",
#           project="incremental_old_model_gfsd_train_m",
#           name='80shots_test_normal_incremental_vedai2dvvedai',
#           imgsz=800,
#           batch=20,
#           device=[0]
#           )