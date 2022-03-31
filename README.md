# Project-FaceMaskDetection
# Đồ án Xác định khuôn mặt đeo khẩu trang
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/pdthuc/Project-FaceMaskDetection)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/pham-dinh-thuc/)
---
## INTRODUCTION

COVID-19 đang ảnh hưởng nặng nề về nhiều mặt trên toàn thế giới. Việc giao tiếp của con người bị ảnh hưởng nhiều, các hệ thống nhận diện truyền thống bị giảm độ chính xác.

Hệ thống của Face Mask Detection sẽ giám sát qua camera, kiểm tra xem người dùng có đeo khẩu trang hay không để thông báo lên màn hình. Trong thực tế chúng ta có thể kết nối ra hệ thống loa để cảnh báo hoặc thông báo cho lực lượng bảo vệ để yêu cầu đeo khẩu trang trước khi vào tòa nhà.

- **Input:** hình ảnh, video hoặc thông qua camera giám sát (CCTV)
- **Output::** Frame của khuôn mặt và Label đã được gán nhãn (Mask/NoMask) để xác định rằng có (hoặc không) đeo khẩu trang.

## DATASETS

Dộ dữ liệu được thu thập từ nhiều nguồn khác nhau:
- [Click to Download](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)

   This dataset consists of 4095 images belonging to two classes:

    - with_mask: 2165 images
    - without_mask: 1930 images
    
- [12k image Kaggle dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
  12K images which are almost 328.92MB in size.

- [VN-Celeb Dataset](https://viblo.asia/p/vn-celeb-du-lieu-khuon-mat-nguoi-noi-tieng-viet-nam-va-bai-toan-face-recognition-Az45bG9VKxY)
  Với hơn 23k khuôn mặt của hơn 1000 người Việt.
 
<ins>**NOTE:**</ins> Trong các bộ dữ liệu ở trên, bộ vn-celeb chỉ có ảnh no_mask thay vì được chia thành 2 bộ (mask và no_mask) như 2 tập còn lại.

Điều này khiến ta cần thêm 1 thuật toán để đeo khẩu trang (mask_the_face) cho các khuôn mặt không đeo khẩu trang.

### [Thuật toán MaskTheFace:](https://github.com/aqeelanwar/MaskTheFace)
  - Thuật toán sẽ dùng 1 máy dò tìm dựa trên DLib để xác định các cạnh và các điểm chính trên khuôn mặt nhằm để xác định các tọa độ qua đó ta có thể tính toán và chèn khẩu trang sao cho tự nhiên.

<ins>DATASET</ins> 
  Chứa khoảng 58 000 images bao gồm:
  - with_mask: ~31 000 images
  - without_mask: ~27 000 images

Ngoài ra, để dữ liệu sát với thực tế, ta còn thu thập dữ liệu từ các cuộc thi tại Việt Nam
  - Zalo AI Challenge 2021 - 5K: Ta sẽ dùng các kĩ thuật xử lý ảnh (image processing) để xác định và cắt các khuôn mặt và gán nhãn cho chúng.
    -  Mask: 6496 images
    -  No Mask: 5889 images
  - Bộ dữ liệu CCTV của FPT Telecom. Ta sẽ dùng các kĩ thuật xử lý ảnh (image processing) để xác định và cắt các khuôn mặt và gán nhãn cho chúng.
    - Mask: 876 images
    - No Mask: 359 images

## ALGORITHMS
- **Images augmentation**
  - Các kĩ thuật được dùng để xử lý ảnh như: Crop, Rotate, Zoom, affine transformations, perspective transformations, contrast changes, gaussian noise, dropout of regions, hue/saturation changes, cropping/padding, blurring, ...
  
   ![alt text](https://github.com/pdthuc/Project-FaceMaskDetection/blob/master/img_src/1.png?raw=true)

  - Các kĩ thuật được sử dụng nhằm làm đa dạng nguồn dữ liệu, tạo nhiều thách thức cho model nhằm tránh Overfit và tăng độ chính xác cho Model khi dự đoán.
  
- **ResNet** (ResNet34, ResNet50)
  - ResNet (Residual Network) được giới thiệu năm 2015. Hiện tại thì có rất nhiều biến thể của kiến trúc ResNet với số lớp khác nhau như ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152,...
  - Mạng ResNet là một mạng CNN được thiết kế để làm việc với hàng trăm hoặc hàng nghìn lớp chập. Một vấn đề xảy ra khi xây dựng mạng CNN với nhiều lớp chập sẽ xảy ra hiện tượng Vanishing Gradient (xảy ra ở Backpropagation – Lan truyền ngược) dẫn tới quá trình học tập không tốt. Mạng ResNet ra đời giải quyết vấn đề đó
  - Resnet sẽ đưa ra các “kết nối tắt” để giúp xuyên qua 1 hay nhiều lớp. Các khối có chức năng như vậy được gọi là Residual Block. 
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    ![alt text](https://github.com/pdthuc/Project-FaceMaskDetection/blob/master/img_src/2.png?raw=true)

  - Mũi tên trong ảnh xuất phát từ đầu và kết thúc tại cuối một khối dư. Nó sẽ bổ sung Input X vào đầy ra của layer. 
  - Tác dụng của việc bổ sung này sẽ giúp chống lại việc đạo hàm bằng 0. Để giá trị dự đoán có được giá trị gần với giá trị thật nhất bằng cách: F(x) + X -> ReLU.
Trong đó, X->weight1->ReLU->weight2.

- **MobileNetV2**
  -  MobileNetV2 là một trong những kiến trúc được ưa chuộng nhất khi phát triển các ứng dụng AI trong computer vision.
  -  MobileNetV2 cũng sử dụng những kết nối tắt như ở mạng ResNet.
  -  Tuy nhiên kết nối tắt ở MobileNetV2 được điều chỉnh sao cho số kênh (hoặc chiều sâu) ở input và output của mỗi block residual được thắt hẹp lại. Chính vì thế nó được gọi là các bottleneck layers (bottleneck là một thuật ngữ thường được sử dụng trong deep learning để ám chỉ các kiến trúc thu hẹp kích thước theo một chiều nào đó).

 **EXPERIMENTS**
   ![Kết quả](https://github.com/pdthuc/Project-FaceMaskDetection/blob/master/img_src/3.png?raw=true)

## IMPROVEMENTS

**Cải thiện phương pháp augmentation:**
- <ins>Mô tả:</ins> Trong dữ liệu tập train, các hình ảnh có kích thước lớn (>100 pixel) khác so với các 
hình ảnh ở tập test (thường nhỏ < 100 pixel, trong khoảng 20 – 50 pixel) dẫn đến việc huấn 
luyện không sát với dữ liệu thực tế.
- Do đó, để cải thiện mô hình, ta sẽ resize các ảnh trong tập train về gần với kích thước ảnh 
trong tập test (khoảng 10 - 40 pixel). Dùng numpy.random.randint(10,40) để chọn ngẫu 
nhiên kích thước cho từng ảnh trong tập train trong khoảng 10 – 40 pixel.
   ![Kết quả](https://github.com/pdthuc/Project-FaceMaskDetection/blob/master/img_src/7.png?raw=true)


 **EXPERIMENTS**
   ![Kết quả](https://github.com/pdthuc/Project-FaceMaskDetection/blob/master/img_src/8.png?raw=true)


## DEPLOY KERAS MODEL WITH TENSORFLOW SERVING

Các bước triển khai Tensorflow Serving:
- Convert tensorflow/keras model (.h5, .cpkt) về định dạng saved_model (.pb).
- Kiểm tra convert model.
- Khởi chạy Tensorflow model server.
- Tiền xử lý dữ liệu và request tới Tensorflow model server.
- Giao tiếp qua RESTful API & gRPC

   ![alt text](https://github.com/pdthuc/Project-FaceMaskDetection/blob/master/img_src/12.png?raw=true)

---
## Tài liệu tham khảo: 
- Face Mask Detection: https://github.com/chandrikadeb7/Face-Mask-Detection
- Mask The Face: https://github.com/aqeelanwar/MaskTheFace
- VN-Celeb dataset: https://viblo.asia/p/vn-celeb-du-lieu-khuon-mat-nguoi-noi-tieng-viet-nam-va-bai-toan-face-recognition-Az45bG9VKxY
- 12k image Kaggle dataset: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
- DLib: http://dlib.net/
- Resnet: https://viblo.asia/p/gioi-thieu-mang-resnet-vyDZOa7R5wj
- MobileNetV2: https://phamdinhkhanh.github.io/2020/09/19/MobileNet.html#:~:text=3.-,MobileNetV2,v%C3%A0%20DeepLabV3%20trong%20image%20segmentation.
- Image augmentation: https://github.com/aleju/imgaug  
- TF-Serving:
  - https://www.tensorflow.org/tfx/guide/serving
  - https://www.tensorflow.org/tfx/guide/serving
- Cuối cùng, xin gửi lời cảm ơn đặc biệt đến [Stack OverFlow](https://stackoverflow.com/) 
