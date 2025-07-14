import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m

#COCO路径
model = YOLO_m("/root/autodl-tmp/COCO_weights/ScaleConvFuse/weights/best.pt")
# print(model.model.modal)
model.model.modal = 'both'

data = 'dv_vedai.yaml'
model.train(data=data,
            epochs=1,
            patience=30,
            batch=2,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="COCO",
            name='coco_transfer',
            exist_ok=True,
            #pretrained=False,
            optimizer='auto',
            seed=0,
            freeze=None,
            #resume=True,
            )
