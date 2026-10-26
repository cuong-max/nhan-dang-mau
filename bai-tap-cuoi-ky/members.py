"""Thành viên nhóm và phân công – dùng chung cho gen_report.py, gen_slides.py, gen_phancong.py."""
MEMBERS = [
    {"name": "Nguyễn Quốc Cường", "mssv": "B26CHKH013",
     "task": "Phân tích bài toán: chọn đề tài và dữ liệu, xác định mục tiêu, thước đo đánh giá; làm slide thuyết trình",
     "short": "Phân tích bài toán, làm slide",
     "products": "Mục 1–2 báo cáo (bài toán, dữ liệu); Slide_NhanDangMau.pptx",
     "share": "50%"},
    {"name": "Trần Như Đức", "mssv": "B26CHKH027",
     "task": "Cài đặt code thuật toán: tiền xử lý, phương án cơ sở và cải tiến, kiểm định chéo, demo; viết báo cáo .docx",
     "short": "Code thuật toán, làm báo cáo",
     "products": "run_experiments.py, robust_dedup.py, app.py; BaoCao_NhanDangMau.docx",
     "share": "50%"},
]
SCHOOL = "HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG"
ADVISOR = "Nguyễn Xuân Đức"
GROUP = " – ".join(m["name"] for m in MEMBERS)
