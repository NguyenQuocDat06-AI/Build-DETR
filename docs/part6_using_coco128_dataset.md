# Hướng dẫn sử dụng bộ dữ liệu COCO128 với mô hình DETR

Bộ dữ liệu `coco128` mà bạn vừa tải về (theo cấu trúc Ultralytics/YOLO) được lưu trữ dưới định dạng **YOLO format**. Tuy nhiên, mô hình DETR theo chuẩn mặc định sử dụng API của **COCO format** (`pycocotools`) để nạp dữ liệu và đánh giá độ chính xác (mAP).

Vì vậy, để tích hợp một cách mượt mà nhất với code DETR mà không phải viết lại toàn bộ luồng Data Augmentation (như Random Crop, Resize padding...), chúng ta sẽ thực hiện 2 bước:
1. Chuyển đổi nhãn (labels) từ định dạng YOLO (.txt) sang định dạng COCO (.json).
2. Tạo DataLoader sử dụng `CocoDetection` của thư viện `torchvision`.

---

## 1. Sự khác biệt giữa 2 định dạng

- **YOLO format**: Mỗi ảnh đi kèm với một file `.txt` tương ứng. Mỗi dòng trong file `.txt` chứa `[class_id, x_center, y_center, width, height]` được **chuẩn hóa (normalize)** trong khoảng từ 0 đến 1 so với kích thước ảnh.
- **COCO format**: Tất cả nhãn của mọi ảnh được gom chung vào một file `instances_train2017.json`. Tọa độ bounding box được lưu dưới dạng giá trị pixel thực tế (absolute) `[x_min, y_min, width, height]`.

---

## 2. Chuyển đổi từ YOLO sang COCO JSON

Tôi đã tạo sẵn cho bạn một file script tên là `yolo2coco.py` nằm trong thư mục `coco128`. Bạn chỉ cần mở terminal và chạy lệnh sau để thực hiện chuyển đổi:

```bash
cd /home/nquocdat06/code/detr/coco128
python yolo2coco.py
```

Sau khi chạy xong, trong thư mục `coco128/annotations` sẽ xuất hiện một file `instances_train2017.json`. File này chứa toàn bộ nhãn của 128 bức ảnh theo chuẩn COCO gốc, giúp DETR đọc được dễ dàng.

Dưới đây là mã nguồn của script `yolo2coco.py` (đã được lưu trong thư mục `coco128`) để bạn tham khảo:

```python
import os
import json
from PIL import Image

def yolo_to_coco(data_dir):
    images_dir = os.path.join(data_dir, 'images/train2017')
    labels_dir = os.path.join(data_dir, 'labels/train2017')
    
    # Tạo thư mục chứa file json
    anno_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Định nghĩa tạm 80 class của COCO
    for i in range(80):
        coco_format["categories"].append({"id": i, "name": f"class_{i}"})
        
    annotation_id = 1
    image_names = sorted(os.listdir(images_dir))
    
    for image_id, image_name in enumerate(image_names):
        img_path = os.path.join(images_dir, image_name)
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            continue
            
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })
        
        label_name = image_name.replace('.jpg', '.txt')
        label_path = os.path.join(labels_dir, label_name)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        
                        # Quy đổi sang giá trị pixel (absolute)
                        abs_cx, abs_cy = cx * width, cy * height
                        abs_w, abs_h = w * width, h * height
                        
                        # (center_x, center_y) -> (x_min, y_min)
                        x_min = abs_cx - (abs_w / 2)
                        y_min = abs_cy - (abs_h / 2)
                        
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, abs_w, abs_h],
                            "area": abs_w * abs_h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                        
    output_path = os.path.join(anno_dir, 'instances_train2017.json')
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"Đã lưu thành công file COCO JSON tại: {output_path}")

if __name__ == "__main__":
    yolo_to_coco('/home/nquocdat06/code/detr/coco128')
```

---

## 3. Cách nạp COCO128 vào mã nguồn DETR

Khi đã có file `.json`, việc đọc dữ liệu hoàn toàn giống với tập COCO chuẩn. Bạn có thể sử dụng `torchvision` trong file khai báo DataLoader của mình:

```python
import torchvision
from torch.utils.data import DataLoader

data_folder = "/home/nquocdat06/code/detr/coco128"
img_folder = f"{data_folder}/images/train2017"
ann_file = f"{data_folder}/annotations/instances_train2017.json"

# Sử dụng module có sẵn của PyTorch
dataset_train = torchvision.datasets.CocoDetection(
    root=img_folder,
    annFile=ann_file,
    transforms=None # Đưa transforms của DETR vào đây
)

# Chú ý: Vì bạn đang dùng RTX 3050 (4GB VRAM)
# Nên hãy set batch_size = 1 hoặc 2 để tránh OOM (Out of memory)
data_loader_train = DataLoader(
    dataset_train,
    batch_size=2, 
    shuffle=True,
    collate_fn=utils.collate_fn # Collate function của DETR
)
```

## 4. Lợi ích của phương pháp này
- **Giữ nguyên source code gốc:** Bạn không phải tự viết lại DataLoader hay sửa đổi cách DETR thực hiện hàm loss `Bipartite Matching`.
- **Dễ đánh giá:** Bạn có thể dùng trực tiếp thư viện `pycocotools` để in ra bảng độ đo mAP, AP50, AP75 giống y hệt như huấn luyện trên tập dataset lớn.
- **Phù hợp máy 4GB VRAM:** Với 128 bức ảnh, vòng lặp epochs sẽ cực nhanh, giúp bạn debug (sanity check) phần code model dễ dàng.
