import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


model = YOLO_m("/root/FoRA/COCO/coco_transfer/weights/best.pt")
# model = YOLO_m("/root/FoRA/model_config/yolov8h_lma_multi_head_ScaleConvFuse.yaml")
# print(model.model.modal)
model.model.modal = 'both'

data = 'dv_vedai.yaml'

model.train(data=data,
            epochs=50,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0,1],
            r_init=9,
            r_target=6,
            adalora=True,
            project="COCO",
            name='AfterTransfer',
            exist_ok=True,
            #pretrained=False,#如果yaml就True，不然False
            optimizer='auto',
            seed=0,
            freeze=None,
            #resume=True,
            )

