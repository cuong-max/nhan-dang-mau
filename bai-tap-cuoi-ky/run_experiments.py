"""Thí nghiệm nhận dạng mẫu: Wine Quality (Vinho Verde, đỏ + trắng) – UCI.
Chạy: python run_experiments.py  -> data/, results/, figures/, model_best.joblib
Bài toán: nhận dạng rượu CHẤT LƯỢNG CAO (điểm cảm quan >= 7, lớp dương=1) từ 11 chỉ tiêu hóa lý + loại rượu.
"""
import json, warnings, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=None, help="thư mục chứa winequality-red.csv, winequality-white.csv (mặc định: ./data, tự tải nếu thiếu)")
SEED = 42; ROOT = Path(__file__).parent; RES, FIG = ROOT / "results", ROOT / "figures"
DAT = Path(ap.parse_args().data_dir).expanduser() if ap.parse_args().data_dir else ROOT / "data"
for p in (RES, FIG, DAT): p.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C_G, C_N = "#7B2D3B", "#3A7CA5"

# ---------------- Dữ liệu ----------------
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
def load(kind):
    f = DAT / f"winequality-{kind}.csv"
    if not f.exists(): urllib.request.urlretrieve(BASE_URL + f.name, f)
    d = pd.read_csv(f, sep=";"); d["is_red"] = int(kind == "red"); return d
raw = pd.concat([load("red"), load("white")], ignore_index=True)
n_dup = int(raw.duplicated().sum())
df = raw.drop_duplicates().reset_index(drop=True)   # loại trùng: tránh rò rỉ train/test (xem robust_dedup.py)
y = (df["quality"] >= 7).astype(int); q = df["quality"]; X = df.drop(columns="quality"); names = list(X.columns)
print("Gốc:", len(raw), "| sau loại trùng:", X.shape, "| chất lượng cao:", y.sum(), f"({y.mean():.1%}) | bản ghi trùng:", n_dup)
df.to_csv(RES / "wine_dataset.csv", index=False)
q.value_counts().sort_index().to_csv(RES / "quality_counts.csv")

# ---------------- Mô hình ----------------
def P(clf, scale=True, pca=None):
    s = [("sc", StandardScaler())] if scale else []
    if pca: s.append(("pca", PCA(n_components=pca, random_state=SEED)))
    return Pipeline(s + [("clf", clf)])
inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
def tuned(pipe, grid, scoring="roc_auc"): return GridSearchCV(pipe, grid, cv=inner, scoring=scoring, n_jobs=-1)
BASE = "Cơ sở: k-NN (k=5, không chuẩn hóa)"
models = {
    "Tham chiếu: luôn đoán lớp đa số": DummyClassifier(strategy="most_frequent"),
    BASE: P(KNeighborsClassifier(5), scale=False),
    "Cơ sở: Naive Bayes Gaussian": P(GaussianNB(), scale=False),
    "k-NN + chuẩn hóa": P(KNeighborsClassifier(15)),
    "LDA": P(LinearDiscriminantAnalysis()),
    "QDA": P(QuadraticDiscriminantAnalysis(reg_param=0.1)),
    "Hồi quy logistic (cân bằng lớp, tuned)": tuned(P(LogisticRegression(max_iter=5000, class_weight="balanced")), {"clf__C": [0.01, 0.1, 1, 10]}),
    "SVM RBF (cân bằng lớp, tuned)": tuned(P(SVC(class_weight="balanced", random_state=SEED)), {"clf__C": [1, 10], "clf__gamma": ["scale", 0.2]}),
    "PCA + SVM RBF (tuned)": tuned(P(SVC(class_weight="balanced", random_state=SEED), pca=8), {"clf__C": [1, 10], "clf__gamma": ["scale", 0.2]}),
    # chọn đặc trưng lọc (ANOVA F) với số đặc trưng k được chọn trong nested CV – theo gợi ý của UCI
    "Chọn đặc trưng (ANOVA, k tuned) + SVM": tuned(Pipeline([("sc", StandardScaler()), ("fs", SelectKBest(f_classif)), ("clf", SVC(class_weight="balanced", random_state=SEED))]),
                                                  {"fs__k": [3, 5, 7, 9, 12], "clf__C": [1, 10]}),
    "Random Forest (cân bằng lớp)": P(RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=1, random_state=SEED, n_jobs=-1), scale=False),
    "Gradient Boosting (HistGB)": P(HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, random_state=SEED), scale=False),
}
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)   # 15 fold
scoring = {"acc": "accuracy", "bacc": "balanced_accuracy", "rec": "recall", "prec": "precision", "f1": "f1", "auc": "roc_auc", "ap": "average_precision"}
rows, fold = [], {}
for n, m in models.items():
    cv = cross_validate(m, X, y, cv=outer, scoring=scoring, n_jobs=1 if isinstance(m, GridSearchCV) else -1)
    fold[n] = cv["test_f1"]
    rows.append({"Mô hình": n, **{k: cv[f"test_{k}"].mean() for k in scoring}, **{k + "_sd": cv[f"test_{k}"].std() for k in scoring}})
    print(f"{n:42s} acc={rows[-1]['acc']:.3f} bacc={rows[-1]['bacc']:.3f} rec={rows[-1]['rec']:.3f} prec={rows[-1]['prec']:.3f} f1={rows[-1]['f1']:.3f} auc={rows[-1]['auc']:.3f}", flush=True)
# Phương án thay thế tận dụng thứ bậc của nhãn: hồi quy SVR trên điểm 3–9, quyết định "cao" nếu điểm dự đoán >= ngưỡng.
# Ngưỡng điểm chọn bằng CV nội bộ trên tập huấn luyện (tối đa F1); cùng 15 fold với các mô hình khác.
ORD = "Hồi quy thứ bậc: SVR điểm → ngưỡng"
def fit_ord(Xtr, qtr):
    ytr = (qtr >= 7).astype(int); sc_in = np.zeros(len(Xtr))
    for a, b_ in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xtr, ytr):
        sc_in[b_] = P(SVR(C=1)).fit(Xtr.iloc[a], qtr.iloc[a]).predict(Xtr.iloc[b_])
    cut = max(np.arange(5.6, 7.01, .05), key=lambda c: f1_score(ytr, (sc_in >= c).astype(int)))
    return P(SVR(C=1)).fit(Xtr, qtr), cut
o = {k: [] for k in scoring}; cuts = []
for tr, te in outer.split(X, y):
    m, cut = fit_ord(X.iloc[tr], q.iloc[tr]); sc_ = m.predict(X.iloc[te]); pr = (sc_ >= cut).astype(int); yt = y.iloc[te]; cuts.append(cut)
    for k, fn in [("acc", accuracy_score), ("bacc", balanced_accuracy_score), ("rec", recall_score), ("prec", precision_score), ("f1", f1_score)]: o[k].append(fn(yt, pr))
    o["auc"].append(roc_auc_score(yt, sc_)); o["ap"].append(average_precision_score(yt, sc_))
fold[ORD] = np.array(o["f1"])
rows.append({"Mô hình": ORD, **{k: np.mean(v) for k, v in o.items()}, **{k + "_sd": np.std(v) for k, v in o.items()}})
print(f"{ORD:42s} f1={rows[-1]['f1']:.3f} auc={rows[-1]['auc']:.3f} rec={rows[-1]['rec']:.3f} prec={rows[-1]['prec']:.3f} ngưỡng TB={np.mean(cuts):.2f}", flush=True)
# Đặc trưng được giữ lại khi chọn đặc trưng trên toàn bộ dữ liệu
fs_full = models["Chọn đặc trưng (ANOVA, k tuned) + SVM"].fit(X, y)
kbest = fs_full.best_params_["fs__k"]; kept = list(X.columns[fs_full.best_estimator_.named_steps["fs"].get_support()])
fscore = pd.Series(f_classif(StandardScaler().fit_transform(X), y)[0], index=names).sort_values(ascending=False); fscore.to_csv(RES / "anova_f.csv")
print("Chọn đặc trưng: k =", kbest, kept)
res = pd.DataFrame(rows).sort_values("f1", ascending=False); res.to_csv(RES / "model_comparison.csv", index=False)

def nb(a, b, n_train, n_test):
    d = a - b; k = len(d); var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var); return d.mean(), t, 2 * stats.t.sf(abs(t), k - 1)
ntr = int(len(X) * .8); nte = len(X) - ntr
sig = [{"Mô hình": n, "Δ F1": (r := nb(fold[n], fold[BASE], ntr, nte))[0], "t": r[1], "p": r[2]} for n in fold if n != BASE]
pd.DataFrame(sig).to_csv(RES / "significance_vs_baseline.csv", index=False)
ord_vs_svm = nb(fold[ORD], fold["SVM RBF (cân bằng lớp, tuned)"], ntr, nte)
json.dump({n: v.tolist() for n, v in fold.items()}, open(RES / "fold_f1.json", "w"))

# ---------------- Dự đoán ngoài-fold ----------------
oof_cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
SVMN = "SVM RBF (cân bằng lớp, tuned)"
svm_params = {k.replace("clf__", ""): v for k, v in models[SVMN].fit(X, y).best_params_.items()}; print("SVM best params:", svm_params)
models_oof = {n: models[n] for n in [BASE, "k-NN + chuẩn hóa", "Hồi quy logistic (cân bằng lớp, tuned)", "Random Forest (cân bằng lớp)", "Gradient Boosting (HistGB)"]}
models_oof[SVMN] = P(SVC(class_weight="balanced", probability=True, random_state=SEED, **svm_params))   # xác suất Platt
oof = {n: cross_val_predict(m, X, y, cv=oof_cv, method="predict_proba")[:, 1] for n, m in models_oof.items()}
# nhãn SVM theo hàm quyết định (ngưỡng 0, giống khi đánh giá CV); xác suất Platt chỉ dùng cho phân tích ngưỡng/demo
svm_pred = cross_val_predict(P(SVC(class_weight="balanced", random_state=SEED, **svm_params)), X, y, cv=oof_cv)
oof_df = pd.DataFrame(oof); oof_df["y"] = y.values; oof_df["quality"] = q.values; oof_df["is_red"] = X["is_red"].values; oof_df.to_csv(RES / "oof_scores.csv", index=False)
# mô hình tốt nhất có xác suất (để demo/ngưỡng)
cand = [n for n in res["Mô hình"] if n in oof and n != BASE]
BEST = cand[0]; print("Mô hình tốt nhất (có xác suất):", BEST)
def pred_at(n, t=None):
    if t is None and n == SVMN: return svm_pred
    s = oof[n]
    if t is None: t = 0.5
    return (s >= t).astype(int)

# Ma trận nhầm lẫn
show = [BASE, "Hồi quy logistic (cân bằng lớp, tuned)", BEST] if BEST != "Hồi quy logistic (cân bằng lớp, tuned)" else [BASE, "SVM RBF (cân bằng lớp, tuned)", BEST]
fig, ax = plt.subplots(1, 3, figsize=(13, 4)); cms = {}
for a, n in zip(ax, show):
    cm = confusion_matrix(y, pred_at(n)); cms[n] = cm.tolist(); a.imshow(cm, cmap="Purples")
    for i in range(2):
        for j in range(2): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_xticks([0, 1], ["Thường", "Chất lượng cao"], fontsize=9); a.set_yticks([0, 1], ["Thường", "Chất lượng cao"], fontsize=9); a.set_xlabel("Dự đoán"); a.set_ylabel("Thực tế"); a.set_title(n, fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=160); plt.close()
json.dump(cms, open(RES / "confusion_matrices.json", "w"), ensure_ascii=False, indent=1)

# ROC + PR
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4)); aucs = {}
for n, s in oof.items():
    fpr, tpr, _ = roc_curve(y, s); aucs[n] = auc(fpr, tpr); ax[0].plot(fpr, tpr, label=f"{n[:34]} ({aucs[n]:.3f})")
    pr, rc, _ = precision_recall_curve(y, s); ax[1].plot(rc, pr)
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC"); ax[0].legend(fontsize=7, loc="lower right")
ax[1].axhline(y.mean(), ls=":", c="k"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision–Recall (đường chấm: tỉ lệ lớp dương)")
plt.tight_layout(); plt.savefig(FIG / "roc_pr.png", dpi=160); plt.close(); json.dump(aucs, open(RES / "oof_auc.json", "w"), ensure_ascii=False, indent=1)

# Boxplot F1 theo fold
order = res["Mô hình"].tolist()[::-1]
fig, ax = plt.subplots(figsize=(8.5, 5.4)); ax.boxplot([fold[n] for n in order], vert=False, labels=order); ax.set_xlabel("F1 (lớp chất lượng cao) trên 15 fold"); plt.tight_layout()
plt.savefig(FIG / "cv_boxplot.png", dpi=160); plt.close()

# ---------------- EDA ----------------
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
for a, f in zip(ax, ["alcohol", "density", "volatile acidity", "chlorides"]):
    lo, hi = X[f].quantile([.005, .995]); b = np.linspace(lo, hi, 30)
    a.hist(X.loc[y == 0, f], bins=b, alpha=.7, color=C_N, label="Thường", density=True); a.hist(X.loc[y == 1, f], bins=b, alpha=.7, color=C_G, label="Chất lượng cao", density=True); a.set_title(f); a.xaxis.set_major_locator(plt.MaxNLocator(4))
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "eda_hist.png", dpi=160); plt.close()
cnt = q.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5.5, 3.6)); ax.bar(cnt.index, cnt.values, color=[C_G if i >= 7 else C_N for i in cnt.index]); ax.set_xlabel("Điểm chất lượng (cảm quan)"); ax.set_ylabel("Số mẫu")
for i, v in zip(cnt.index, cnt.values): ax.text(i, v + 30, v, ha="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "quality_dist.png", dpi=160); plt.close()
corr = X.drop(columns="is_red").corr(); fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(11), corr.columns, rotation=90, fontsize=8); ax.set_yticks(range(11), corr.columns, fontsize=8); plt.colorbar(im, fraction=.046); plt.tight_layout(); plt.savefig(FIG / "eda_corr.png", dpi=160); plt.close()
Xs = StandardScaler().fit_transform(X); pf = PCA().fit(Xs); cum = np.cumsum(pf.explained_variance_ratio_); Z = PCA(2).fit_transform(Xs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].scatter(Z[y == 0, 0], Z[y == 0, 1], s=5, c=C_N, alpha=.35, label="Thường"); ax[0].scatter(Z[y == 1, 0], Z[y == 1, 1], s=7, c=C_G, alpha=.6, label="Chất lượng cao")
ax[0].set_xlabel(f"PC1 ({pf.explained_variance_ratio_[0]:.1%})"); ax[0].set_ylabel(f"PC2 ({pf.explained_variance_ratio_[1]:.1%})"); ax[0].legend(markerscale=3)
ax[1].plot(range(1, 13), cum, "o-"); ax[1].axhline(.95, ls=":", c="gray"); ax[1].set_xlabel("Số thành phần chính"); ax[1].set_ylabel("Phương sai tích lũy"); plt.tight_layout(); plt.savefig(FIG / "pca.png", dpi=160); plt.close()
pc95 = int(np.argmax(cum >= .95) + 1)

# ---------------- Phân tích lỗi theo điểm chất lượng gốc ----------------
pb = oof_df.copy(); pb["pred"] = pred_at(BEST)
byq = pb.groupby("quality").apply(lambda g: pd.Series({"n": len(g), "tỉ lệ dự đoán 'cao'": g["pred"].mean()}))
byq.to_csv(RES / "pred_rate_by_quality.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.8)); ax.bar(byq.index, byq["tỉ lệ dự đoán 'cao'"], color=[C_G if i >= 7 else C_N for i in byq.index]); ax.set_xlabel("Điểm chất lượng thực tế"); ax.set_ylabel("Tỉ lệ mô hình dự đoán 'chất lượng cao'")
for i, v in zip(byq.index, byq["tỉ lệ dự đoán 'cao'"]): ax.text(i, v + .02, f"{v:.0%}", ha="center", fontsize=9)
ax.set_ylim(0, 1.1); plt.tight_layout(); plt.savefig(FIG / "pred_by_quality.png", dpi=160); plt.close()
bt = {}
for nm, msk in [("Đỏ", X["is_red"] == 1), ("Trắng", X["is_red"] == 0)]:
    yy, pp, ss = y[msk], pb.loc[msk.values, "pred"], oof[BEST][msk.values]
    bt[nm] = {"n": int(msk.sum()), "pos": float(yy.mean()), "f1": f1_score(yy, pp), "rec": recall_score(yy, pp), "prec": precision_score(yy, pp), "auc": roc_auc_score(yy, ss)}
json.dump(bt, open(RES / "by_wine_type.json", "w"), ensure_ascii=False, indent=1)
# Lỗi: z-score trung bình theo nhóm TP/FN/FP/TN
grp = np.select([(y == 1) & (pb["pred"] == 1), (y == 1) & (pb["pred"] == 0), (y == 0) & (pb["pred"] == 1)], ["TP", "FN", "FP"], "TN")
z = (X - X.mean()) / X.std(); prof = z.groupby(grp).mean().T[["TP", "FN", "FP", "TN"]]; prof.round(2).to_csv(RES / "error_profile.csv")
pd.Series(grp, name="nhóm").value_counts().to_csv(RES / "error_counts.csv")

# Ngưỡng tối ưu F1 / recall mục tiêu
s = oof[BEST]; ths = np.linspace(.05, .9, 86); f1s = [f1_score(y, (s >= t).astype(int)) for t in ths]
fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.plot(ths, f1s, label="F1")
ax.plot(ths, [precision_score(y, (s >= t).astype(int), zero_division=0) for t in ths], "--", label="Precision"); ax.plot(ths, [recall_score(y, (s >= t).astype(int)) for t in ths], ":", label="Recall")
tb = ths[int(np.argmax(f1s))]; ax.axvline(tb, c="gray", lw=.8); ax.set_xlabel("Ngưỡng xác suất 'chất lượng cao'"); ax.legend(); plt.tight_layout(); plt.savefig(FIG / "threshold.png", dpi=160); plt.close()
thr = []
for t in [.5, float(tb), .3]:
    p_ = (s >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, p_).ravel()
    thr.append({"Ngưỡng": round(float(t), 2), "TP": int(tp), "FP": int(fp), "FN": int(fn), "Precision": tp / (tp + fp), "Recall": tp / (tp + fn), "F1": f1_score(y, p_)})
pd.DataFrame(thr).to_csv(RES / "threshold_tradeoff.csv", index=False)
tall = []
for n, sc in oof.items():
    f = [f1_score(y, (sc >= t).astype(int)) for t in ths]
    tall.append({"Mô hình": n, "F1 @0,5": f1_score(y, (sc >= .5).astype(int)), "F1 tốt nhất": max(f), "Ngưỡng": float(ths[int(np.argmax(f))]), "AUC": roc_auc_score(y, sc)})
pd.DataFrame(tall).sort_values("F1 tốt nhất", ascending=False).to_csv(RES / "threshold_all_models.csv", index=False)

# ---------------- Độ quan trọng & độ nhạy ----------------
rf = RandomForestClassifier(500, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False); imp.to_csv(RES / "rf_importance.csv")
fig, ax = plt.subplots(figsize=(6.3, 4.4)); imp[::-1].plot.barh(ax=ax, color=C_G); ax.set_xlabel("Độ quan trọng (Random Forest)"); plt.tight_layout(); plt.savefig(FIG / "importance.png", dpi=160); plt.close()
fr = [.05, .1, .2, .4, .7, 1.0]; lc = []
for f in fr:
    v = []
    for r in range(5):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=r)
        if f < 1: Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=f, stratify=ytr, random_state=r)
        rfm = RandomForestClassifier(300, class_weight="balanced_subsample", random_state=r, n_jobs=-1).fit(Xtr, ytr)
        kn = P(KNeighborsClassifier(5), scale=False).fit(Xtr, ytr)
        sv = P(SVC(class_weight="balanced", random_state=r, **svm_params)).fit(Xtr, ytr)
        v.append((roc_auc_score(yte, sv.decision_function(Xte)), roc_auc_score(yte, kn.predict_proba(Xte)[:, 1]), roc_auc_score(yte, rfm.predict_proba(Xte)[:, 1])))
    lc.append(np.mean(v, 0))
lc = np.array(lc); ks = [1, 2, 3, 5, 8, 12]; kf = []
for k in ks:
    top = list(imp.index[:k]); kf.append(roc_auc_score(y, cross_val_predict(P(SVC(class_weight="balanced", random_state=SEED, **svm_params)), X[top], y, cv=oof_cv, method="decision_function")))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, kf, "o-"); ax[0].set_xlabel("Số đặc trưng (top theo RF)"); ax[0].set_ylabel("AUC (SVM RBF)")
ax[1].plot(fr, lc[:, 0], "o-", label="SVM RBF"); ax[1].plot(fr, lc[:, 2], "^-", label="Random Forest"); ax[1].plot(fr, lc[:, 1], "s--", label="Cơ sở: k-NN thô"); ax[1].set_xlabel("Tỉ lệ dữ liệu huấn luyện"); ax[1].set_ylabel("AUC (test)"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG / "sensitivity.png", dpi=160); plt.close()
hi = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack(); hi = hi[hi.abs() > .5]
json.dump({"ks": ks, "k_auc": kf, "frac": fr, "lc_svm": lc[:, 0].tolist(), "lc_knn": lc[:, 1].tolist(), "lc_rf": lc[:, 2].tolist(), "pc95": pc95, "cum": cum.tolist(), "best": BEST,
           "n": len(X), "n_raw": len(raw), "fs_k": int(kbest), "fs_kept": kept, "ord_cut": float(np.mean(cuts)), "ord_vs_svm": {"dF1": float(ord_vs_svm[0]), "t": float(ord_vs_svm[1]), "p": float(ord_vs_svm[2])}, "pos": int(y.sum()), "n_dup": n_dup, "svm_params": {k: str(v) for k, v in svm_params.items()}, "thr_best": float(tb), "high_corr": {f"{a} ~ {b}": float(v) for (a, b), v in hi.items()}},
          open(RES / "summary.json", "w"), ensure_ascii=False, indent=1)

# Mô hình cho demo
final = (models_oof[SVMN] if BEST == SVMN else P(RandomForestClassifier(500, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1), scale=False) if "Forest" in BEST else
         P(HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, random_state=SEED), scale=False) if "Boosting" in BEST else
         P(LogisticRegression(max_iter=5000, class_weight="balanced", C=1))).fit(X, y)
joblib.dump({"model": final, "features": names, "name": BEST, "thr": float(tb)}, ROOT / "model_best.joblib"); print("Xong.")
