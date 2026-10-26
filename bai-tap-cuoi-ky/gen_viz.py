"""Minh họa động cách các mô hình phân biệt rượu chất lượng cao, trên 2 đặc trưng mạnh nhất (độ cồn, axit bay hơi).
Sinh viz/viz_data.js (dữ liệu cho viz/PhanLop_Dong.html) và figures/phan_lop_dong.gif."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

FX, FY = "alcohol", "volatile acidity"
df = pd.read_csv("results/wine_dataset.csv")
X, y = df[[FX, FY]].to_numpy(), (df["quality"] >= 7).astype(int).to_numpy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
real = pd.read_csv("results/model_comparison.csv").set_index("Mô hình")

sig = lambda z: 1 / (1 + np.exp(-z))
MODELS = [  # (khóa, tên hiển thị, mô hình, hàm điểm, tên trong bảng kết quả 12 đặc trưng, ý tưởng)
    ("knn_raw", "Cơ sở: k-NN (k=5, không chuẩn hóa)", KNeighborsClassifier(5), None, "Cơ sở: k-NN (k=5, không chuẩn hóa)",
     "Bỏ phiếu theo 5 láng giềng gần nhất. Không chuẩn hóa nên độ cồn (thang 8–15) lấn át axit bay hơi (thang 0,1–1,6): biên gần như chỉ cắt theo trục ngang."),
    ("knn", "k-NN + chuẩn hóa", make_pipeline(StandardScaler(), KNeighborsClassifier(15)), None, "k-NN + chuẩn hóa",
     "Chuẩn hóa đưa hai đặc trưng về cùng thang, láng giềng được tính theo cả hai chiều. Biên lởm chởm vì phụ thuộc vào từng điểm cục bộ."),
    ("nb", "Cơ sở: Naive Bayes Gaussian", GaussianNB(), None, "Cơ sở: Naive Bayes Gaussian",
     "Mỗi lớp là một phân phối chuẩn, coi các đặc trưng độc lập. Biên là đường cong bậc hai, ưu tiên lớp đa số nên bỏ sót nhiều rượu tốt."),
    ("lr", "Hồi quy logistic (cân bằng lớp)", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")), None,
     "Hồi quy logistic (cân bằng lớp, tuned)",
     "Biên là một đường thẳng. Cân bằng lớp tăng trọng số cho lớp thiểu số (19% mẫu) nên biên dịch về phía rượu thường, bắt được nhiều rượu tốt hơn."),
    ("svm", "SVM RBF (cân bằng lớp) – mô hình chọn", make_pipeline(StandardScaler(), SVC(C=1, gamma=0.5, class_weight="balanced")), "dec",
     "SVM RBF (cân bằng lớp, tuned)",
     "Nhân RBF uốn biên theo hình dạng dữ liệu; cân bằng lớp giữ recall cao. Đây là mô hình được chọn trong báo cáo (12 đặc trưng)."),
    ("rf", "Random Forest (cân bằng lớp)", RandomForestClassifier(300, min_samples_leaf=10, class_weight="balanced", random_state=0, n_jobs=-1), None,
     "Random Forest (cân bằng lớp)",
     "Trung bình phiếu của 300 cây quyết định, mỗi cây chia theo ngưỡng trục ngang/dọc, nên biên có dạng bậc thang."),
]

x0, x1, y0, y1, NX, NY = 8.0, 14.6, 0.08, 1.2, 96, 64
gx, gy = np.linspace(x0, x1, NX), np.linspace(y0, y1, NY)
G = np.array([[a, b] for b in gy for a in gx])

def score(m, kind, Z):
    return sig(2 * m.decision_function(Z)) if kind == "dec" else m.predict_proba(Z)[:, 1]

out = {"fx": "Độ cồn (% thể tích)", "fy": "Axit bay hơi (g/L)", "ext": [x0, x1, y0, y1], "nx": NX, "ny": NY,
       "n_train": int(len(ytr)), "n_test": int(len(yte)), "pos_rate": float(y.mean()), "models": []}
rng = np.random.default_rng(1)
keep = rng.choice(len(yte), 900, replace=False)  # điểm hiển thị, lấy từ tập kiểm tra
out["pts"] = [[round(float(a), 2), round(float(b), 3), int(c)] for (a, b), c in zip(Xte[keep], yte[keep])]
grids = {}
for key, name, m, kind, rname, idea in MODELS:
    m.fit(Xtr, ytr)
    g = score(m, kind, G).reshape(NY, NX); grids[key] = g
    s_te = score(m, kind, Xte)
    r = real.loc[rname]
    out["models"].append({"key": key, "name": name, "idea": idea,
                          "grid": [int(v) for v in np.round(g.ravel() * 100)],
                          "s": [int(round(v * 100)) for v in s_te[keep]],
                          "test_s": [int(round(v * 100)) for v in s_te], "real": {k: round(float(r[k]), 3) for k in ["f1", "rec", "prec", "bacc", "auc"]}})
    print(name, "F1 2D:", round(float(((s_te >= .5) & (yte == 1)).sum() * 2 / ((s_te >= .5).sum() + yte.sum())), 3))
out["test_y"] = [int(v) for v in yte]

Path("viz").mkdir(exist_ok=True)
Path("viz/viz_data.js").write_text("window.VIZ = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")

# ---------- GIF: biên quyết định chuyển dần giữa các mô hình ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
HI, LO = "#9B2C45", "#2272B4"
cmap = LinearSegmentedColormap.from_list("q", ["#C9DDF0", "#F7F3F2", "#EFC9D2"])
pts = Xte[keep]; lab = yte[keep]
fig, ax = plt.subplots(figsize=(8, 5.4), dpi=90)
fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.17)
keys = [m[0] for m in MODELS]; names = [m[1] for m in MODELS]
HOLD, MORPH = 14, 10
frames = [(i, j / MORPH) for i in range(len(keys)) for j in list(range(MORPH)) + [MORPH] * HOLD]

def draw(f):
    i, t = f; a, b = grids[keys[i - 1]] if i else grids[keys[0]], grids[keys[i]]
    e = t * t * (3 - 2 * t); g = a * (1 - e) + b * e
    ax.clear()
    ax.imshow(g, origin="lower", extent=[x0, x1, y0, y1], aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.contour(gx, gy, g, levels=[.5], colors="#2B2024", linewidths=2)
    ax.scatter(*pts[lab == 0].T, s=9, c=LO, alpha=.55, lw=0, label="Rượu thường (< 7 điểm)")
    ax.scatter(*pts[lab == 1].T, s=16, c=HI, alpha=.9, lw=.6, edgecolors="white", label="Chất lượng cao (≥ 7 điểm)")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("Độ cồn (% thể tích)"); ax.set_ylabel("Axit bay hơi (g/L)")
    ax.legend(loc="upper right", fontsize=8, framealpha=.9)
    fig.suptitle("Mô hình phân biệt rượu vang chất lượng cao như thế nào?", fontsize=13, fontweight="bold", x=.09, ha="left")
    ax.set_title(f"{i + 1}/{len(keys)}  {names[i]}" + ("" if t >= 1 else "  …"), loc="left", fontsize=10.5, color="#5A1E2B")

fig.text(.09, .02, "Vùng hồng: dự đoán chất lượng cao · đường đen: biên quyết định (ngưỡng 0,5) · 900 mẫu tập kiểm tra",
         fontsize=8, color="#4A3F43")
FuncAnimation(fig, draw, frames=frames).save("figures/phan_lop_dong.gif", writer=PillowWriter(fps=8))
print("ok", len(frames), "frames")
