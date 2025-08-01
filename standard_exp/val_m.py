import ultralytics.global_mode as gb
gb.set_global_mode('fuse')
from ultralytics import YOLO_m

model = YOLO_m('/root/FoRA/COCO/AfterTransfer/weights/best.pt')
# print(model.model.modal)
model.model.modal = 'fuse'
# print(model.model.modal)

data = 'new_dv_vedai.yaml'
model.val(data=data,
          project='dv_vedai',
          name='test_train',
          imgsz=800,
          batch=30,
          device=[0,1]
          # visualize=True,
          # split='val_rgb',
          )
