# README.md

### Data Preparation

**SUN-RGBD**

OV-Uni3DETR에서 제공하는 train/valid [pkl](https://drive.google.com/drive/folders/1ljh6quUw5gLyHbQiY68HDGtY6QLp_d6e)사용(ovuni3detr_sunrgbd_annos/sunrgbd_infos_train_pls_ens_10c36c.pkl, sunrgbd_infos_val_withimg.pkl). SUN-RGBD의 label은 v1과 v2가 있으며  OV-Uni3DETR에서 제공하는 pkl은 v2 기반. tools/label_v2_to_v1.py 으로 v2기반 pkl을 v1으로 만들 수 있음(코드 내 디렉토리 변경필요). 

**ScanNet**

TBD

**Clip Embed**

config에 pts_bbox_head/zeroshot_path='sunrgbd_clip_a+cname_rn50_manyprompt_46c_coda.npy' 은 CLIP기반으로 class에 대한 text feature을 추출한것. CoDA에서 제공하는 [링크](https://github.com/yangcaoai/CoDA_NeurIPS2023/blob/main/CLIP/notebooks/Prompt_Engineering_for_ImageNet.ipynb)를 통해 원하는 클래스를 넣어서 제작 가능

---

### **Training**

|  | (v2)AP_n | AP_b | AP_all | (v1)AP_n | AP_b | AP_all |
| --- | --- | --- | --- | --- | --- | --- |
| [Baseline](https://gitlab.lgresearch.ai/visionlab/intern/yc_ov3dod/ov3dod/-/blob/main/projects/configs/ours_sunrgbd/ours_sunrgbd_pc.py?ref_type=heads) | 9.11 | 48.4 | 18.25 | 8.7 | 47.47 | 17.12 |
| [Ours](https://gitlab.lgresearch.ai/visionlab/intern/yc_ov3dod/ov3dod/-/blob/main/projects/configs/ours_sunrgbd/ours_sunrgbd_pc_dn_saf_ray.py?ref_type=heads) | 11.22 | 50.4 | 20.34 | 10.18 | 49.33 | 18.69 |

```bash
# Single GPU
python ./extra_tools/train.py ./projects/configs/ours_sunrgbd/ours_sunrgbd_pc.py --work-dir ./work_dirs/ours
# Multi GPU
bash ./extra_tools/dist_train.sh ./projects/configs/ours_sunrgbd/ours_sunrgbd_pc.py 8 --work-dir ./work_dirs/ours
```
