import ultralytics.global_mode as gb #一定要这么引用，不要from
gb.set_global_mode('both')
from ultralytics import YOLO_m


#model = YOLO_m("/root/FoRA/yolov8n_lma_multi_head.yaml")
student = YOLO_m("/root/FoRA/yolov8n_lma_multi_head.yaml")
teacher = YOLO_m("/root/autodl-tmp/multi_head_yolov8n_lma_weights/weights/best.pt")

# print(model.model.modal)
student.model.modal = 'both'
teacher.model.modal = 'both'

student.train(data="/root/FoRA/data_yaml/vedaifewshot_dv_vedai.yaml",
            teacher=teacher.model,
            distill_layers=[6,8,21],
            distillation_loss="cwd",
            epochs=1,
            patience=10,
            batch=1,
            imgsz=800,
            device=[0],
            r_init=9,
            r_target=6,
            adalora=True,
            project="distill_train_m",
            name='test_distill',
            # pretrained=False,
            optimizer='SGD',
            # lr0 = 0.2,
            seed=0,
            freeze=55,
            # resume=True,
            )
