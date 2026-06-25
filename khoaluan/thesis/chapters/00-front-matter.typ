= Lời cam đoan

Chúng tôi cam đoan khóa luận tốt nghiệp ngành Thị giác máy tính với đề tài "Phân lớp bệnh tim đa phương thức trên ảnh điện tâm đồ (ECG): từ hỏi–đáp thị giác của PULSE sang bộ phân lớp theo tác vụ trên PTB-XL" là công trình khoa học do chúng tôi thực hiện dưới sự hướng dẫn của PGS.TS Lý Quốc Ngọc.

Những kết quả nghiên cứu của khóa luận hoàn toàn trung thực và chính xác. Mọi tham khảo từ công trình của người khác đều được trích dẫn đầy đủ.

#v(1em)
#align(right)[Sinh viên thực hiện \ (Ký tên, họ tên)]

= Lời cảm ơn

Chúng tôi xin gửi lời cảm ơn chân thành đến PGS.TS Lý Quốc Ngọc đã tận tình hướng dẫn, định hướng và góp ý trong suốt quá trình thực hiện khóa luận. Chúng tôi cũng xin cảm ơn quý Thầy Cô Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG.HCM đã truyền đạt kiến thức nền tảng; cảm ơn các tác giả của bộ dữ liệu PTB-XL và mô hình PULSE-7B đã công khai tài nguyên phục vụ nghiên cứu; và cảm ơn gia đình, bạn bè đã luôn động viên, hỗ trợ.

#v(1em)
#align(right)[TP. Hồ Chí Minh, tháng #box(width: 1.2cm, repeat[.]) năm 2026 \ Sinh viên thực hiện]

= Trang thông tin khóa luận

*Tên đề tài:* Phân lớp bệnh tim đa phương thức trên ảnh điện tâm đồ (ECG): từ hỏi–đáp thị giác của PULSE sang bộ phân lớp theo tác vụ trên PTB-XL.

*Ngành:* Thị giác máy tính.

*Họ tên sinh viên thực hiện:* Nguyễn Huỳnh Hải Đăng (22127052 — nhhdang22\@clc.fitus.edu.vn); Phan Vũ Gia Hân (22127102 — pvghan22\@clc.fitus.edu.vn).

*Khóa đào tạo:* 2022–2026.

*Giảng viên hướng dẫn:* PGS.TS Lý Quốc Ngọc.

*Cơ sở đào tạo:* Trường Đại học Khoa học Tự nhiên, ĐHQG.HCM.

== Tóm tắt nội dung khóa luận

Khóa luận nghiên cứu bài toán phân lớp bệnh tim đa nhãn từ ảnh điện tâm đồ (ECG) trên bộ dữ liệu PTB-XL, sử dụng nền tảng mô hình ngôn ngữ–thị giác PULSE-7B (kiến trúc họ LLaVA-v1.6-Vicuna-7B). Bản chất PULSE là một mô hình hỏi–đáp thị giác (VQA) sinh văn bản chứ chưa phải một bộ phân lớp đa nhãn được hiệu chuẩn xác suất; do đó mục tiêu của khóa luận là chuyển PULSE thành một bộ phân lớp bệnh tim đa nhãn có khả năng phân biệt nhãn mạnh mà không phá vỡ lõi mô hình. Khóa luận phân rã bài toán PTB-XL thành ba tác vụ độc lập (chẩn đoán, nhịp, hình thái), thiết kế quy tắc gán nhãn hai tầng tôn trọng đặc thù dữ liệu, giữ nguyên hàm mất mát entropy chéo next-token nguyên gốc của PULSE và chỉ tinh chỉnh nhẹ mô hình ngôn ngữ bằng QLoRA 4-bit trên nhánh thị giác đông cứng. Một đóng góp về phương pháp đánh giá là giao thức teacher-forced: tự dựng khối nhãn và đọc xác suất tại các vị trí token đã biết trong một lượt suy luận, cho phép đo macro-AUC không phụ thuộc ngưỡng một cách nhất quán và đầy đủ trên mọi lớp — thay cho đánh giá sinh tự do vốn không trích được điểm ở khoảng 73% mẫu trong một thử nghiệm ban đầu. Trên tập kiểm thử độc lập fold-10 của PTB-XL, mô hình (tinh chỉnh 2 epoch) đạt macro-AUC trung bình 0.876 (chẩn đoán 0.893, nhịp 0.887, hình thái 0.849), và vượt chính mô hình nền PULSE-7B chưa tinh chỉnh thêm trung bình +0.097 macro-AUC trong một so sánh có kiểm soát cùng giao thức. Khi đưa về đúng phân nhóm Super 5 siêu lớp mà PULSE đánh giá, mô hình đề xuất đạt macro-AUC 0.902 và macro-F1 0.693 trên fold-10. Kết quả cho thấy một mô hình ngôn ngữ–thị giác có thể phân biệt bệnh tim đa nhãn từ ảnh ECG ở mức tốt mà không cần đến tín hiệu một chiều gốc; do macro-AUC chỉ phản ánh khả năng phân biệt, đây chưa phải khẳng định về một bộ phân lớp đã hiệu chuẩn sẵn sàng lâm sàng.

== Những kết quả mới của khóa luận

Khóa luận có bốn đóng góp chính: (1) tái định dạng bài toán PTB-XL theo ba tác vụ độc lập kèm quy tắc gán nhãn hai tầng (chẩn đoán theo ngưỡng likelihood, nhịp và hình thái theo sự hiện diện), khắc phục lỗi loại bỏ nhầm nhóm nhịp/hình thái; (2) đề xuất giao thức đánh giá teacher-forced cho phép đo macro-AUC không phụ thuộc ngưỡng một cách nhất quán và tái lập trên mọi lớp, thay cho đánh giá sinh tự do thiếu tin cậy (không trích được điểm ở khoảng 73% mẫu trong thử nghiệm ban đầu); (3) thiết lập so sánh có kiểm soát với chính mô hình nền PULSE-7B trên cùng tập kiểm thử và cùng giao thức, qua đó tách bạch đóng góp của bước tinh chỉnh (vượt +0.097 macro-AUC); (4) ghi nhận và phân tích một thử nghiệm ban đầu thất bại làm cơ sở cho việc chuyển hướng thiết kế. Trên fold-10, phương pháp đạt macro-AUC trung bình 0.876, cao hơn rõ rệt các backbone tổng quát ở mức gần ngẫu nhiên (GPT-4o 0.556, LLaVA-1.6 0.500; lưu ý khác phân nhóm nhãn, xem Mục 4.2.2) và vượt chính mô hình nền PULSE-7B trên cùng tập nhãn (+0.097).

== Các ứng dụng / khả năng ứng dụng / những vấn đề còn bỏ ngỏ

Phương pháp đặt nền móng cho công cụ hỗ trợ đọc ECG đa nhãn trực tiếp từ ảnh, hữu ích khi chỉ có ảnh điện tâm đồ (không có tín hiệu số gốc) như trong nhiều tình huống lâm sàng thực tế. Các vấn đề còn bỏ ngỏ gồm: bổ sung năng lực sinh lập luận (reasoning) cho chẩn đoán; thu hẹp khoảng cách với baseline đơn phương thức trên tín hiệu; huấn luyện thêm để khai thác dư địa còn lại; và mở rộng đánh giá đa fold để khẳng định khả năng tổng quát hóa.

#v(1.5em)
#grid(columns: (1fr, 1fr), align: center,
  [*GIẢNG VIÊN HƯỚNG DẪN* \ (Ký tên, họ tên)],
  [*SINH VIÊN THỰC HIỆN* \ (Ký tên, họ tên)],
)
#v(1em)
#align(center)[*XÁC NHẬN CỦA CƠ SỞ ĐÀO TẠO* \ *HIỆU TRƯỞNG*]

= Thesis information

*Thesis title:* Multimodal cardiac disease classification on ECG images: from PULSE's visual question answering to a task-decomposed classifier on PTB-XL.

*Speciality:* Computer Vision.

*Name of Students:* Nguyen Huynh Hai Dang (22127052 — nhhdang22\@clc.fitus.edu.vn); Phan Vu Gia Han (22127102 — pvghan22\@clc.fitus.edu.vn).

*Academic year:* 2022–2026.

*Supervisor:* Assoc. Prof. Dr. Ly Quoc Ngoc.

*At:* VNUHCM – University of Science.

== Summary

This thesis studies multi-label cardiac disease classification from electrocardiogram (ECG) images on the PTB-XL dataset, built on the PULSE-7B vision–language model (LLaVA-v1.6-Vicuna-7B family). PULSE is fundamentally a generative visual-question-answering model rather than a probability-calibrated multi-label classifier; the goal is therefore to turn it into a multi-label classifier with strong label-discrimination ability without modifying its core. We decompose the PTB-XL problem into three independent tasks (diagnostic, rhythm, form), design a two-tier labeling rule that respects the dataset's likelihood convention, keep PULSE's native next-token cross-entropy objective, and only lightly fine-tune the language model with 4-bit QLoRA while freezing the vision branch. A methodological contribution is the teacher-forced evaluation protocol, which builds the label block and reads probabilities at known token positions in a single forward pass, enabling consistent, threshold-free macro-AUC scoring on every class and replacing free-form generation that failed to produce a score on about 73% of samples in an initial experiment. On the held-out fold-10 test set, the model reaches a mean macro-AUC of 0.876 (diagnostic 0.893, rhythm 0.887, form 0.849), and surpasses the untouched PULSE-7B base model by +0.097 mean macro-AUC under a controlled, identical protocol. On PULSE's own PTB-XL Super subgroup (5 superclasses), the proposed model attains a macro-AUC of 0.902 and a macro-F1 of 0.693 on fold-10.

== Novelty of thesis

(1) A task-decomposed reformulation of PTB-XL with a two-tier labeling rule; (2) a teacher-forced evaluation protocol enabling consistent, threshold-free macro-AUC on every class; (3) a controlled comparison against the base PULSE-7B on the same test set and protocol, isolating the fine-tuning gain (+0.097 macro-AUC); (4) an analysis of an initial failed experiment that motivated the design pivot.

== Applications / applicability / perspective

The method lays groundwork for tools that read multi-label ECGs directly from images, useful when only the ECG image is available. Open problems include adding grounded reasoning generation, closing the gap to single-modality signal baselines, mitigating overfitting on the rhythm task, and multi-fold evaluation.

= Danh mục các ký hiệu, các chữ viết tắt

#table(
  columns: (auto, 1fr),
  inset: 6pt,
  align: (left, left),
  table.header([*Viết tắt*], [*Diễn giải*]),
  [AUC], [Diện tích dưới đường ROC (Area Under the ROC Curve)],
  [CE], [Hàm mất mát entropy chéo (Cross-Entropy)],
  [CLIP], [Contrastive Language–Image Pre-training],
  [ECG], [Tín hiệu điện tâm đồ (Electrocardiogram)],
  [LLM], [Mô hình ngôn ngữ lớn (Large Language Model)],
  [LoRA], [Low-Rank Adaptation],
  [QLoRA], [Quantized Low-Rank Adaptation],
  [SCP-ECG], [Standard Communications Protocol for Computer-Assisted Electrocardiography (mã chú giải PTB-XL)],
  [VLM], [Mô hình ngôn ngữ–thị giác (Vision–Language Model)],
  [VQA], [Hỏi–đáp thị giác (Visual Question Answering)],
)
