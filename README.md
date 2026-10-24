# Tuyển tập bài tập — Học máy / Khai phá dữ liệu

Kho chứa các bài tập cá nhân. Mỗi bài nằm trong một thư mục `bai-tap-*` riêng,
dùng chung một môi trường ảo `venv/` ở thư mục gốc.

## Cấu trúc tổng quan

```
nhan-dang-mau/
├─ bai-tap-1/                                  # Bài tập 1 (đã hoàn thành)
│  ├─ src/pipeline_regression.py               #   code chạy thực nghiệm
│  ├─ outputs/                                 #   kết quả: CSV, JSON, figures/*.png
│  └─ BaoCao_TieuLuan_PCA_HoiQuy_TiengViet.docx#   báo cáo (tiếng Việt)
├─ bai-tap-2/                                  # Bài tập 2 (đã hoàn thành)
│  ├─ logistic_regression.ipynb                #   notebook (sản phẩm nộp chính)
│  ├─ src/                                     #   build_notebook.py, generate_report_vi.py
│  ├─ outputs/                                 #   summary.json, figures/*.png
│  └─ BaoCao_TieuLuan_LogisticRegression_TiengViet.docx
├─ bai-tap-cuoi-ky/                            # Bài tập nhóm cuối kỳ — Nhận dạng mẫu
│  ├─ run_experiments.py, robust_dedup.py      #   code thí nghiệm
│  ├─ app.py                                   #   demo Streamlit
│  ├─ data/  results/  figures/                #   dữ liệu UCI, kết quả, hình
│  ├─ BaoCao_NhanDangMau.docx / .pdf           #   báo cáo
│  ├─ Slide_NhanDangMau.pptx                   #   slide thuyết trình
│  └─ PhanCong_TuDanhGia.docx                  #   phân công & tự đánh giá
├─ venv/                                       # môi trường ảo dùng chung
├─ requirements.txt                            # thư viện dùng chung
├─ .gitignore                                  # bỏ qua venv/ khi push git
└─ README.md
```

## Cài đặt môi trường (một lần, dùng chung cho mọi bài)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Bài tập 1 — Pipeline hồi quy với tiền xử lý và giảm chiều PCA

**Đề:** Trên một bộ dữ liệu có target liên tục, xây dựng chương trình dự báo,
so sánh baseline với các mô hình hồi quy và kiểm tra tác động của PCA.
Chọn mô hình dựa trên MAE/RMSE, R² và độ ổn định qua Cross-Validation. (Làm cá nhân.)

- **Dữ liệu:** Iris (UCI ML Repository, ID 53) — 150 mẫu, 4 đặc trưng.
- **Lưu ý:** Iris vốn là bài toán *phân loại*. Vì đề yêu cầu **target liên tục**,
  bài làm chuyển thành **hồi quy**: dự báo `petal width (cm)` (liên tục) từ 3 đặc trưng còn lại.
- **Pipeline:** `PolynomialFeatures(2) → StandardScaler → [PCA(0.95)] → Model`
  (toàn bộ tiền xử lý nằm *trong* pipeline ⇒ không rò rỉ dữ liệu khi CV).
- **Baseline:** `DummyRegressor` (dự báo trung bình).
  **Mô hình so sánh:** Linear, Ridge, Lasso, KNN, DecisionTree, RandomForest,
  GradientBoosting, SVR(RBF) — mỗi mô hình chạy 2 phiên bản (có / không PCA).

**Chạy:**

```powershell
python bai-tap-1\src\pipeline_regression.py
```

Sinh lại toàn bộ kết quả trong `bai-tap-1/outputs/` (CSV, JSON, 6 biểu đồ).

**Kết quả chính:**

| | RMSE | R² |
|---|---|---|
| Baseline (mean) | 0.803 | −0.005 |
| **RandomForest (No-PCA)** — tốt nhất | **0.186 ± 0.028** | **0.935 ± 0.020** |

- Mọi mô hình vượt xa baseline (R² từ ≈0 lên ≈0.94).
- PCA giảm chiều mạnh (9 → 2 chiều giữ 95% phương sai) nhưng **làm giảm nhẹ độ chính xác**
  trên bộ dữ liệu nhỏ, sạch này — một kết luận hợp lệ: PCA đánh đổi độ chính xác lấy
  số chiều nhỏ hơn / tốc độ / chống quá khớp, không phải lúc nào cũng cải thiện kết quả.

> Ghi chú: mã nguồn (comment, thông báo, nhãn biểu đồ) viết bằng tiếng Anh;
> báo cáo `.docx` viết bằng tiếng Việt.

---

## Bài tập 2 — Cài đặt và đánh giá Logistic Regression (NumPy vs scikit-learn)

**Đề:** Trên một bộ dữ liệu thực tế tự chọn, tự cài đặt Logistic Regression bằng NumPy
(không dùng model/optimizer có sẵn) và so sánh với `sklearn.linear_model.LogisticRegression`.
**Sản phẩm nộp: một Jupyter notebook chạy được từ đầu đến cuối.** (Làm cá nhân.)

- **Dữ liệu:** Breast Cancer Wisconsin (UCI) — 569 mẫu, 30 đặc trưng, 2 lớp (ác tính 212 /
  lành tính 357). Nạp offline qua sklearn nên notebook chạy được ở mọi máy, không cần tải.
- **Cài đặt NumPy:** class `LogisticRegressionNumpy` — sigmoid, score, log loss, gradient
  descent cập nhật `w` & `b` (η=0.1, ≤5000 vòng, dừng khi |Δlog loss|<1e-8, L2 λ=0.01).
  Vẽ log loss trên train theo vòng lặp.
- **So sánh công bằng:** sklearn dùng cùng mức L2 (đặt `C = 1/(m·λ)`) ⇒ cùng hàm mục tiêu.
- **Đánh giá:** bảng accuracy, precision, recall, F1, ROC-AUC, log loss + ma trận nhầm lẫn +
  ROC. ROC-AUC và log loss tính từ xác suất dự đoán.

**Chạy notebook:**

```powershell
jupyter notebook bai-tap-2\logistic_regression.ipynb
# hoặc chạy toàn bộ không cần mở giao diện:
jupyter nbconvert --to notebook --execute --inplace bai-tap-2\logistic_regression.ipynb
```

> Lưu ý: cần cài thêm Jupyter (`pip install jupyter nbconvert nbformat ipykernel`) — đã có
> trong môi trường nếu bạn cài từ `requirements.txt` sau cập nhật.

**Kết quả chính (hai bản gần như trùng khớp — minh chứng bản NumPy cài đúng):**

| Chỉ số | NumPy (tự cài) | scikit-learn |
|---|---|---|
| Accuracy | 0.9790 | 0.9790 |
| Precision | 1.0000 | 1.0000 |
| Recall | 0.9434 | 0.9434 |
| F1 | 0.9709 | 0.9709 |
| ROC-AUC | 0.9983 | 0.9983 |
| Log loss | 0.0753 | 0.0753 |

- Trọng số hai bản tương quan ≈ 1.0, bias trùng nhau, cùng 3 mẫu bị phân loại sai.
- Gradient descent hội tụ sau ~4690 vòng; với hàm lồi, bộ tối ưu chỉ ảnh hưởng đường đi
  chứ không ảnh hưởng nghiệm.

> Ghi chú: notebook (markdown + code) viết bằng tiếng Anh; báo cáo `.docx` viết bằng tiếng Việt.

**Sinh lại báo cáo `.docx`** (sau khi đã chạy notebook để có `outputs/`):

```powershell
python bai-tap-2\src\generate_report_vi.py
```

---

## Bài tập nhóm cuối kỳ — Nhận dạng rượu vang chất lượng cao (Wine Quality)

**Đề:** Mỗi nhóm thực hiện một bài toán nhận dạng mẫu trên dữ liệu thực tế; có phương án cơ sở,
ít nhất một phương án cải tiến, phân tích lỗi, hạn chế; nộp báo cáo, slide, mã nguồn, bảng phân công; điểm cộng cho demo.

- **Dữ liệu:** Wine Quality (UCI, ID 186, Cortez et al. 2009) — rượu đỏ 1.599 + trắng 4.898;
  loại 1.177 bản ghi trùng → 5.320 mẫu, 11 chỉ tiêu hóa lý + loại rượu.
- **Bài toán:** nhận dạng rượu **chất lượng cao** (điểm cảm quan ≥ 7, chiếm 19%).
- **Cơ sở:** k-NN trên dữ liệu thô, Naive Bayes. **Cải tiến:** chuẩn hóa, cân bằng lớp, SVM RBF, logistic,
  LDA/QDA, RF, Gradient Boosting, chọn đặc trưng, hồi quy thứ bậc (SVR), chọn ngưỡng.
- **Đánh giá:** CV phân tầng 5×3 lồng nhau (nested), t-test hiệu chỉnh Nadeau–Bengio.

**Kết quả chính:**

| Mô hình | F1 (lớp chất lượng cao) | Recall | AUC |
|---|---|---|---|
| Cơ sở: k-NN thô | 0.277 | 0.210 | 0.692 |
| SVM RBF + chuẩn hóa + cân bằng lớp | 0.548 | 0.809 | 0.848 |
| **Hồi quy thứ bậc SVR → ngưỡng** | **0.570** | 0.690 | 0.852 |

- Phần lớn lỗi nằm ở ranh giới điểm 6/7; bản ghi trùng làm thổi phồng F1 của Random Forest (0.665 → 0.450 khi loại trùng).

**Chạy** (từ thư mục `bai-tap-cuoi-ky`):

```powershell
cd bai-tap-cuoi-ky
python run_experiments.py      # ~10 phút → results/, figures/, model_best.joblib
python robust_dedup.py
python gen_report.py; python gen_slides.py
streamlit run app.py           # demo tại http://localhost:8501
```

Chi tiết: [`bai-tap-cuoi-ky/README.md`](bai-tap-cuoi-ky/README.md).
