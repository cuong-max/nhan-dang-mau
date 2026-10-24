"""Demo tương tác: chạy `streamlit run app.py` (cần chạy run_experiments.py trước).
Chọn mẫu rượu có sẵn hoặc kéo 11 chỉ tiêu hóa lý -> xem dự đoán của mô hình cơ sở và mô hình cải tiến."""
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st, joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent

@st.cache_resource
def load():
    df = pd.read_csv(ROOT / "results" / "wine_dataset.csv")
    X, y = df.drop(columns="quality"), (df["quality"] >= 7).astype(int)
    best = joblib.load(ROOT / "model_best.joblib")
    knn = KNeighborsClassifier(5).fit(X, y)                       # cơ sở: không chuẩn hóa
    pca = make_pipeline(StandardScaler(), PCA(2)).fit(X)
    return df, X, y, best, knn, pca, pca.transform(X)

df, X, y, best, knn, pca, Z = load()
feats = best["features"]

st.set_page_config(page_title="Nhận dạng rượu vang chất lượng cao", layout="wide")
st.title("Nhận dạng rượu vang chất lượng cao (điểm ≥ 7)")
st.caption("Dữ liệu Wine Quality – Cortez et al. (2009), UCI. Mô hình huấn luyện trên toàn bộ dữ liệu nên mẫu có sẵn là dự đoán trong-mẫu (lạc quan).")

sb = st.sidebar
idx = sb.number_input("Chọn mẫu có sẵn (chỉ số 0–%d)" % (len(df) - 1), 0, len(df) - 1, 0)
if sb.button("Lấy ngẫu nhiên 1 mẫu chất lượng cao"):
    st.session_state["rand"] = int(np.random.choice(np.where(y == 1)[0]))
if "rand" in st.session_state and sb.checkbox("Dùng mẫu ngẫu nhiên vừa lấy", True):
    idx = st.session_state["rand"]
thr = sb.slider("Ngưỡng quyết định (xác suất 'chất lượng cao')", 0.05, 0.95, round(best["thr"], 2), 0.01,
                help=f"Mặc định = ngưỡng tối ưu F1 ({best['thr']:.2f}). Hạ ngưỡng -> bắt nhiều rượu ngon hơn nhưng báo nhầm nhiều hơn.")
base = X.iloc[int(idx)]
st.markdown(f"**Mẫu {int(idx)}** – rượu {'đỏ' if base['is_red'] else 'trắng'}, điểm thực tế: **{int(df.loc[int(idx), 'quality'])}**")

x = base.copy()
cols = st.columns(4)
for i, f in enumerate([c for c in feats if c != "is_red"]):
    lo, hi = float(X[f].quantile(.01)), float(X[f].quantile(.99))
    x[f] = cols[i % 4].slider(f, min(lo, float(base[f])), max(hi, float(base[f])), float(base[f]), key=f"{f}{idx}")
x["is_red"] = 1 if cols[3].radio("Loại rượu", ["trắng", "đỏ"], index=int(base["is_red"]), key=f"t{idx}") == "đỏ" else 0

xf = x[feats].to_frame().T.astype(float)
p_best = best["model"].predict_proba(xf)[0, 1]
p_knn = knn.predict_proba(xf)[0, 1]
c1, c2 = st.columns(2)
for col, name, p, t in [(c1, "Cơ sở: k-NN (không chuẩn hóa)", p_knn, 0.5), (c2, f"Cải tiến: {best['name']}", p_best, thr)]:
    col.subheader(name); col.metric("Xác suất chất lượng cao", f"{p:.1%}", help=f"ngưỡng {t:.2f}")
    (col.success if p >= t else col.info)("Dự đoán: CHẤT LƯỢNG CAO" if p >= t else "Dự đoán: THƯỜNG")
if abs(p_best - thr) < 0.1:
    st.warning("Mẫu nằm sát ngưỡng – giống nhóm rượu điểm 6/7 mà mô hình hay nhầm.")

l, r = st.columns(2)
with l:
    st.subheader("Vị trí mẫu trong không gian PCA 2D")
    zp = pca.transform(xf)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, c="#3A7CA5", alpha=.3, label="Thường")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=6, c="#7B2D3B", alpha=.5, label="Chất lượng cao")
    ax.scatter(zp[:, 0], zp[:, 1], marker="*", s=350, c="gold", edgecolors="k", label="Mẫu đang xét")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=2, fontsize=8); st.pyplot(fig)
with r:
    st.subheader("Ảnh hưởng của nồng độ cồn")
    grid = np.linspace(X["alcohol"].quantile(.01), X["alcohol"].quantile(.99), 40)
    rows = pd.concat([xf] * len(grid), ignore_index=True); rows["alcohol"] = grid
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(grid, best["model"].predict_proba(rows)[:, 1]); ax.axhline(thr, ls=":", c="gray"); ax.axvline(x["alcohol"], c="gold")
    ax.set_xlabel("alcohol (% vol)"); ax.set_ylabel("P(chất lượng cao)"); ax.set_ylim(0, 1); st.pyplot(fig)

st.subheader("Dự đoán hàng loạt từ file CSV")
up = st.file_uploader("CSV có các cột: " + ", ".join(feats) + " (dấu phân cách ',' hoặc ';')", type="csv")
if up:
    d = pd.read_csv(up, sep=None, engine="python")
    if "is_red" not in d.columns: d["is_red"] = 0
    miss = [f for f in feats if f not in d.columns]
    if miss: st.error(f"Thiếu cột: {miss}")
    else:
        d["P(chất lượng cao)"] = best["model"].predict_proba(d[feats])[:, 1]
        d["Dự đoán"] = np.where(d["P(chất lượng cao)"] >= thr, "Chất lượng cao", "Thường")
        st.dataframe(d); st.download_button("Tải kết quả", d.to_csv(index=False).encode(), "du_doan.csv")
