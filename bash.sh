CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./extra_tools/dist_train.sh ./projects/configs/ov_uni3detr/ov_uni3detr_sunrgbd_pc_agn_46cls.py 4 --work-dir ./work_dirs/ov_uni3detr_sunrgbd_pc_agn_46cls
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./extra_tools/dist_train.sh ./projects/configs/ov_uni3detr/ov_uni3detr_sunrgbd_pc_agn.py 4 --work-dir ./work_dirs/ov_uni3detr_sunrgbd_pc_agn_top10cls
CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./extra_tools/dist_train.sh ./projects/configs/ov_uni3detr/ov_uni3detr_sunrgbd_pc.py 4 --work-dir ./work_dirs/ov_uni3detr_sunrgbd_pc_46cls
