"""Kiểm tra độ bền: loại bản ghi trùng (rò rỉ giữa train/test) rồi đánh giá lại. -> results/dedup_check.csv"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
# đọc dữ liệu GỐC (chưa loại trùng) từ data/ – results/wine_dataset.csv đã được loại trùng
df = pd.concat([pd.read_csv(f"data/winequality-{k}.csv", sep=";").assign(is_red=int(k == "red")) for k in ("red", "white")], ignore_index=True)
d2 = df.drop_duplicates(); print(len(df), "->", len(d2))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42); rows = []
M = {"Cơ sở: k-NN thô": Pipeline([("c", KNeighborsClassifier(5))]),
     "SVM RBF (C=10, cân bằng lớp)": Pipeline([("s", StandardScaler()), ("c", SVC(C=10, class_weight="balanced"))]),
     "Random Forest (cân bằng lớp)": Pipeline([("c", RandomForestClassifier(300, class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
     "Gradient Boosting (HistGB)": Pipeline([("c", HistGradientBoostingClassifier(learning_rate=.08, max_iter=300, random_state=42))])}
for tag, d in [("Gốc (6497)", df), (f"Đã loại trùng ({len(d2)})", d2)]:
    X, y = d.drop(columns="quality"), (d["quality"] >= 7).astype(int)
    for n, m in M.items():
        r = cross_validate(m, X, y, cv=cv, scoring={"f1": "f1", "auc": "roc_auc", "rec": "recall", "prec": "precision"}, n_jobs=-1)
        rows.append({"Dữ liệu": tag, "Mô hình": n, "F1": r["test_f1"].mean(), "AUC": r["test_auc"].mean(), "Recall": r["test_rec"].mean(), "Precision": r["test_prec"].mean()})
pd.DataFrame(rows).to_csv("results/dedup_check.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
