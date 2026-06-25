= Mở đầu

== Giới thiệu

Điện tâm đồ (ECG) là công cụ chẩn đoán tim mạch phổ biến, chi phí thấp và không xâm lấn. PTB-XL là một trong những bộ dữ liệu ECG công khai lớn và được chú giải kỹ nhất, phát hành công khai trên PhysioNet @wagner, truy cập #link("https://physionet.org/content/ptb-xl/1.0.3/")[tại đây]. Mỗi bản ghi ECG trong PTB-XL được gán nhiều phát biểu SCP, với tổng cộng 71 mã chia thành 44 mã chẩn đoán, 12 mã nhịp và 19 mã hình thái, trong đó có 4 mã chồng lấn giữa nhóm chẩn đoán và hình thái @wagner. Benchmark chuẩn của Strodthoff và cộng sự vận hành PTB-XL như nhiều tác vụ riêng biệt với split cố định: 8 fold huấn luyện, fold 9 kiểm định và fold 10 kiểm thử @strodthoff.

Gần đây, các mô hình ngôn ngữ–thị giác (VLM) được áp dụng mạnh vào y tế. PULSE-7B là một mô hình chuyên biệt cho ảnh ECG, dạy kiến trúc họ LLaVA-v1.6-Vicuna-7B đọc hiểu ảnh điện tâm đồ qua tập chỉ dẫn ECGInstruct @pulse. Tuy nhiên, PULSE bản chất là một mô hình hỏi–đáp thị giác (VQA) sinh văn bản, chưa phải một bộ phân lớp đa nhãn được hiệu chuẩn xác suất. Khóa luận này tập trung vào việc chuyển PULSE thành một bộ phân lớp bệnh tim đa nhãn có khả năng phân biệt nhãn mạnh trên PTB-XL mà không phá vỡ lõi mô hình.

== Động lực nghiên cứu

=== Ý nghĩa khoa học

Việc dùng một mô hình sinh ngôn ngữ–thị giác làm bộ phân lớp đa nhãn đặt ra hai câu hỏi: làm thế nào để định dạng bài toán cho khớp với cơ chế sinh token, và làm thế nào để đánh giá xác suất đa nhãn một cách đáng tin cậy khi đầu ra là văn bản tự do. Khóa luận đóng góp một cách tái cấu trúc bài toán theo tác vụ và một giao thức đánh giá teacher-forced giải quyết hai câu hỏi đó, có giá trị tham chiếu cho các nghiên cứu dùng VLM/LLM đa phương thức làm bộ phân lớp.

=== Ý nghĩa thực tiễn

Trong nhiều tình huống lâm sàng, dữ liệu sẵn có chỉ là ảnh in/chụp của điện tâm đồ chứ không phải tín hiệu số gốc. Một mô hình phân lớp bệnh tim đa nhãn hoạt động trực tiếp trên ảnh ECG do đó có khả năng hỗ trợ sàng lọc và giảm tải cho bác sĩ, đặc biệt ở các cơ sở thiếu công cụ đọc tín hiệu số.

== Phát biểu bài toán

Cho một ảnh điện tâm đồ và một câu hỏi tác vụ dạng văn bản, mô hình cần dự đoán tập nhãn bệnh đa nhãn tương ứng cùng xác suất cho từng lớp. Bài toán PTB-XL được tách thành ba tác vụ độc lập: chẩn đoán (44 mã), nhịp (12 mã) và hình thái (19 mã). Bài toán được phát biểu hình thức như sau.

*Đầu vào.* Với mỗi mẫu cần phân lớp, đầu vào gồm hai luồng dữ liệu và một từ vựng mã theo tác vụ:

- Ảnh ECG: $I in bb(R)^(H times W times 3)$ — tín hiệu 12 chuyển đạo được kết xuất thành ảnh RGB rồi đưa qua bộ mã hóa thị giác CLIP ViT-L/14 ở độ phân giải 336.
- Câu hỏi tác vụ: $q_t$ — chuỗi token văn bản (tiếng Anh) mô tả tác vụ $t in \{$ chẩn đoán, nhịp, hình thái $\}$.
- Từ vựng mã của tác vụ: $V_t = \{c_1, dots.h, c_(K_t)\}$, với $K_("chẩn đoán") = 44$, $K_("nhịp") = 12$, $K_("hình thái") = 19$.

*Đầu ra.* Với mỗi mã $c_k in V_t$, mô hình ước lượng xác suất mã đó hiện diện trên ảnh:

#grid(columns: (1fr, auto, 1fr), align: horizon,
  [], $p_k = P(y_k = 1 thin | thin I, q_t) in [0, 1]$, align(right)[(1.1)],
)

từ đó thu được vector nhãn nhị phân sau khi áp ngưỡng theo từng lớp $tau_k$:

#grid(columns: (1fr, auto, 1fr), align: horizon,
  [], $hat(y)_k = bb(1)[p_k >= tau_k]$, align(right)[(1.2)],
)

trong đó $bb(1)[dot]$ là hàm chỉ thị (bằng 1 nếu điều kiện trong ngoặc đúng, ngược lại bằng 0). Như vậy mỗi tác vụ là một bài phân lớp đa nhãn trên $K_t$ mã, tức một ánh xạ $f_theta : (I, q_t) arrow.r.bar (p_1, dots.h, p_(K_t)) in [0, 1]^(K_t)$. Khác với một bộ phân lớp truyền thống, xác suất $p_k$ KHÔNG lấy từ một đầu sigmoid rời ngoài kiến trúc mà được đọc trực tiếp từ phân phối token của mô hình sinh tại vị trí nhãn của mã $c_k$ theo giao thức teacher-forced (Mục 3.6). Đầu ra có thể mở rộng kèm một khối lập luận (reasoning) đặt trước khối nhãn, song phần lập luận được để ngỏ như hướng phát triển.

== Thách thức bài toán

Bài toán có ba thách thức chính. Thứ nhất, PULSE là mô hình sinh, nên ở chế độ sinh tự do đầu ra có thể không trích được điểm số hoặc rơi vào trạng thái lặp, sinh khối nhãn hỏng. Thứ hai, phân phối lớp của PTB-XL mất cân bằng nặng (lớp NORM chiếm khoảng 44% số mẫu) @ptbxlimb, dễ đẩy mô hình tới hiện tượng sụp đổ chế độ (luôn dự đoán lớp đa số). Thứ ba, giá trị likelihood của PTB-XL chỉ có ý nghĩa cho nhóm chẩn đoán; nhóm nhịp và hình thái bị gán likelihood bằng 0, nên một ngưỡng likelihood áp đồng loạt sẽ loại bỏ nhầm toàn bộ hai nhóm này @wagner.

== Phạm vi nghiên cứu

Khóa luận giới hạn ở: bộ dữ liệu PTB-XL; mô hình nền PULSE-7B (không tiền huấn luyện lại nhánh thị giác); tinh chỉnh nhẹ bằng QLoRA trên mô hình ngôn ngữ; và đánh giá phân lớp đa nhãn theo ba tác vụ bằng macro-AUC (chỉ số chính) cùng macro-F1 (tham khảo). Phần sinh lập luận được để ngỏ như hướng phát triển.

== Bố cục luận văn

Khóa luận được tổ chức thành 5 chương. Chương 1 trình bày phần mở đầu. Chương 2 trình bày tổng quan cơ sở lý thuyết và các công trình liên quan. Chương 3 trình bày phương pháp nghiên cứu đề xuất. Chương 4 trình bày kết quả thực nghiệm và phân tích. Chương 5 đưa ra kết luận và hướng phát triển.
