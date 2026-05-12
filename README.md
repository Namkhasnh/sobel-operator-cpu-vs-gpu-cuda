# 🖼️ Sobel Edge Detection: CPU vs GPU (CUDA)

> Phát hiện biên ảnh bằng bộ lọc Sobel — so sánh hiệu năng giữa CPU tuần tự và GPU song song với CUDA.

---

## 📌 Giới thiệu

**Phát hiện biên (Edge Detection)** là bước tiền xử lý cơ bản trong thị giác máy tính, giúp xác định vùng thay đổi cường độ pixel đột ngột — tức là biên của vật thể trong ảnh. Bộ lọc **Sobel** là một trong những phương pháp phổ biến nhất, ổn định và hiệu quả cho nhiệm vụ này.

Dự án này triển khai bộ lọc Sobel theo **hai hướng**:

| Phiên bản | Công nghệ | File |
|-----------|-----------|------|
| CPU (tuần tự) | C++ + OpenCV | `sobel_opencv.cpp` |
| GPU (song song) | CUDA + OpenCV | `sobel_cuda.cu` |

Mục tiêu chính là **so sánh tốc độ xử lý** giữa hai cách tiếp cận và minh họa lợi ích của tính toán song song trên GPU.

---

## 🖼️ Demo

| Input (ảnh gốc) | Output (Sobel Edge) |
|:-:|:-:|
| ![Input](input2.jpg) | ![Output](output22.png) |
| Ảnh phong cảnh núi & hồ màu sắc tự nhiên | Biên được phát hiện — đường viền núi, sóng nước nổi bật rõ ràng trên nền đen |

> Bộ lọc Sobel bắt trọn biên của dãy núi Teton, đường chân trời phản chiếu trên mặt hồ, và kết cấu sóng nước — minh chứng rõ nét cho khả năng phát hiện gradient cường độ pixel.

---

## 🧠 Kỹ thuật & Bài toán

### Bộ lọc Sobel là gì?

Sobel là một **bộ lọc đạo hàm xấp xỉ** — nó tính gradient của ảnh theo hai hướng ngang (Gx) và dọc (Gy), sau đó kết hợp để cho ra độ lớn gradient tại mỗi điểm ảnh.

**Ma trận kernel Sobel:**

```
Wx (ngang):          Wy (dọc):
[ -1  0  1 ]         [ -1  -2  -1 ]
[ -2  0  2 ]         [  0   0   0 ]
[ -1  0  1 ]         [  1   2   1 ]
```

**Công thức tính gradient:**

```
Gx = Wx ⊛ I      (tích chập theo chiều ngang)
Gy = Wy ⊛ I      (tích chập theo chiều dọc)

Magnitude = sqrt(Gx² + Gy²)
```

Kết quả là ảnh grayscale, trong đó pixel **sáng = biên rõ**, pixel **tối = vùng đồng nhất**.

### Thách thức hiệu năng

Với ảnh kích thước `W × H`, phép tích chập Sobel cần duyệt qua **toàn bộ W×H pixel**, mỗi pixel tính tổng **9 phép nhân + cộng** từ vùng lân cận 3×3. Đây là bài toán **data-parallel** điển hình — mỗi pixel hoàn toàn độc lập, lý tưởng để song song hóa trên GPU.

---

## ⚙️ Cách tính toán song song được áp dụng

### Phiên bản CPU (tuần tự)

```
Vòng lặp y (0 → H):
  Vòng lặp x (0 → W):
    Tính Gx, Gy cho pixel (x, y)
    Ghi output[y][x]
```

Xử lý **từng pixel một**, hoàn toàn tuần tự → phụ thuộc vào một lõi CPU.

---

### Phiên bản CUDA (song song hàng loạt)

```
Mỗi thread GPU xử lý một pixel (x, y) độc lập:
  Thread (tx, ty) ↔ Pixel (x, y)
  Tính Gx, Gy từ constant memory
  Ghi d_output[y * width + x]
```

**Kiến trúc luồng:**

```
Grid (2D)
├── Block (8×8 threads)
│   ├── Thread (0,0) → pixel (0,0)
│   ├── Thread (1,0) → pixel (1,0)
│   ├── ...
│   └── Thread (7,7) → pixel (7,7)
├── Block (1,0) → pixel (8..15, 0..7)
└── ...
```

**Cấu hình launch:**

```cuda
dim3 blockSize(8, 8);         // 64 threads/block
dim3 gridSize(
    (W + 7) / 8,
    (H + 7) / 8
);
sobel_kernel_gpu<<<gridSize, blockSize>>>(...);
```

**Các tối ưu hóa CUDA được sử dụng:**

| Kỹ thuật | Mô tả |
|----------|-------|
| `__constant__ memory` | Ma trận Wx, Wy lưu trong constant memory — mọi thread đọc đồng thời không tốn băng thông global |
| `__restrict__` pointer | Gợi ý compiler không có aliasing → tối ưu hóa instruction |
| 2D thread block | Block 8×8 phản ánh đúng cấu trúc 2D của ảnh, tăng locality |
| CUDA Events | Đo thời gian chính xác bằng `cudaEventRecord` thay vì `chrono` CPU |
| Clamp biên | Xử lý pixel biên bằng clamp (không zero-pad) → không cần kiểm tra phức tạp |

---

## 📋 Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|------------|---------|
| GPU | NVIDIA GPU hỗ trợ CUDA (Compute Capability ≥ 3.5) |
| CUDA Toolkit | ≥ 10.0 |
| OpenCV | ≥ 4.x (`libopencv-dev`) |
| Compiler | `g++` (C++14+), `nvcc` |
| OS | Linux (Ubuntu khuyến nghị) / Google Colab |

---

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường (trên Ubuntu / Google Colab)

```bash
apt-get update -qq
apt-get install -y libopencv-dev
```

### 2. Chuẩn bị ảnh đầu vào

Đặt ảnh đầu vào (`.jpg`, `.png`, ...) vào thư mục hiện tại. Ví dụ: `input.jpg`.

---

### 3. Build & chạy phiên bản CPU

```bash
# Biên dịch
g++ sobel_opencv.cpp -o sobel_opencv `pkg-config --cflags --libs opencv4` -O2

# Chạy
./sobel_opencv input.jpg output_cpu.png
```

**Output mẫu:**
```
Loaded image: 1920 x 1080, channels = 3
CPU Sobel time: 245.83 ms
Saved output to: output_cpu.png
```

---

### 4. Build & chạy phiên bản GPU (CUDA)

```bash
# Kiểm tra GPU
nvidia-smi

# Biên dịch
nvcc sobel_cuda.cu -o sobel_cuda `pkg-config --cflags --libs opencv4` -O2

# Chạy
./sobel_cuda input.jpg output_gpu.png
```

**Output mẫu:**
```
Image size: 1920 x 1080, channels = 3
GPU Sobel execution time: 3.21 ms
Output saved to: output_gpu.png
```

---

## 📊 Benchmark

Kết quả đo trên **Google Colab T4 GPU** với ảnh `input2.jpg`:

| Phiên bản | Thời gian xử lý | Ghi chú |
|-----------|----------------|---------|
| CPU (`sobel_opencv`) | ~245 ms | C++, O2, 1 lõi CPU |
| GPU (`sobel_cuda`) | ~3 ms | CUDA, block 8×8, T4 GPU |
| **Speedup** | **~80×** | GPU nhanh hơn ~80 lần |

### Lý giải speedup

| Yếu tố | CPU | GPU |
|--------|-----|-----|
| Số luồng xử lý | 1 | 2.560+ (CUDA cores T4) |
| Phương thức | Tuần tự pixel | Song song toàn bộ ảnh |
| Bộ nhớ kernel | Cache L1/L2 | Constant memory (broadcast) |
| Thời gian tính | O(W×H) thực sự tuần tự | O(1) về mặt lý thuyết |

> **Lưu ý:** Thời gian GPU bao gồm `cudaMemcpy` (host → device → host). Nếu tính riêng kernel execution, tốc độ còn cao hơn nữa.

---

## 🗂️ Cấu trúc dự án

```
.
├── sobel_opencv.cpp     # Phiên bản CPU (C++ + OpenCV)
├── sobel_cuda.cu        # Phiên bản GPU (CUDA + OpenCV)
├── input.jpg            # Ảnh đầu vào mẫu
├── output_cpu.png       # Ảnh kết quả từ CPU
├── output_gpu.png       # Ảnh kết quả từ GPU
└── README.md
```

---

## 🔍 Chi tiết Implementation

### CPU (`sobel_opencv.cpp`)

```cpp
void apply_sobel_cpu(const Mat& input, Mat& output) {
    // 1. Chuyển ảnh sang grayscale
    cvtColor(input, gray, COLOR_BGR2GRAY);

    // 2. Duyệt từng pixel với padding biên (clamp)
    for (int y = 0; y < gray.rows; ++y)
        for (int x = 0; x < gray.cols; ++x) {
            // Tích chập 3x3 với Wx và Wy
            double mag = sqrt(gx*gx + gy*gy);
            output.at<uchar>(y, x) = min(mag, 255.0);
        }
}
```

### GPU (`sobel_cuda.cu`)

```cuda
__global__ void sobel_kernel_gpu(
    const uint8_t* d_input,
    uint8_t* d_output,
    int width, int height, int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Mỗi thread tính độc lập một pixel
    // Đọc Wx, Wy từ constant memory
    double Gx = 0, Gy = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            // Clamp biên + tính gray nội tuyến
            Gx += d_Wx[ky+1][kx+1] * gray;
            Gy += d_Wy[ky+1][kx+1] * gray;
        }

    d_output[y*width+x] = (uint8_t)min(sqrt(Gx*Gx+Gy*Gy), 255.0);
}
```

---

## 💡 Mở rộng có thể thực hiện

- [ ] Dùng **shared memory** để cache tile ảnh, giảm truy cập global memory
- [ ] Thử block size khác nhau (16×16, 32×32) và so sánh
- [ ] Áp dụng **Canny Edge Detection** (Sobel + NMS + Hysteresis) trên CUDA
- [ ] Benchmark trên nhiều kích thước ảnh (512×512, 1080p, 4K)
- [ ] So sánh với `cv::Sobel()` built-in của OpenCV (đã tối ưu)

---

## 📚 Tài liệu tham khảo
 
[1] M. Rakhimov, M. Ochilov, S. Javliev, and R. Nasimov, "CUDA Block Size Optimization for Gaussian and Sobel Filters: Benchmarking Against CPU Implementations," in *Proceedings of the International Conference on Future Networks and Distributed Systems (ICFNDS '25)*, Dubai, United Arab Emirates, Dec. 2025. doi: [10.1145/3789692.3789772](https://doi.org/10.1145/3789692.3789772)
 
[2] J. J. Tse, "Image Processing with CUDA," M.S. thesis, School of Comput. Sci., Univ. of Nevada, Las Vegas, NV, USA, Aug. 2012. doi: [10.34917/4332680](https://doi.org/10.34917/4332680)
 
[3] M. K. Hossain, M. A. Ashique, M. A. Ibtehaz, and J. Uddin, "Parallel Edge Detection using Sobel Algorithm with Contract-time Anytime Algorithm in CUDA," Dept. Comput. Sci. and Eng., BRAC University, Dhaka, Bangladesh.
 
[4] K. A. Spiridonov, I. S. Stulov, and I. A. Ferapontov, "Implementation and comparison of the Sobel operator on CPU and GPU using CUDA," *International Journal of Humanities and Natural Sciences*, vol. 10-5, no. 97, pp. 66–69, 2024. doi: [10.24412/2500-1000-2024-10-5-66-69](https://doi.org/10.24412/2500-1000-2024-10-5-66-69)
 
[5] A. Smailović, "Time performance comparison between GPU and CPU computing by implementing an edge-detection algorithm using Sobel filter," *Bulletin of the Transilvania University of Braşov, Series I: Engineering Sciences*, vol. 12 (61), no. 1, 2019. doi: [10.31926/but.ens.2019.12.61.7](https://doi.org/10.31926/but.ens.2019.12.61.7)
 
