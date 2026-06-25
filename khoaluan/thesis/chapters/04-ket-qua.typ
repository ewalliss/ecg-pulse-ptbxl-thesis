= Kết quả nghiên cứu và các phân tích, đánh giá, thảo luận

== Thiết lập thực nghiệm

=== Mục tiêu thực nghiệm

Thực nghiệm nhằm trả lời: (i) mô hình có thực sự phân biệt nhãn hay sụp đổ về lớp đa số; (ii) hiệu năng so với mô hình nền PULSE-7B và các mốc tham chiếu; (iii) khả năng tổng quát hóa từ tập kiểm định sang tập kiểm thử.

=== Dữ liệu thực nghiệm và công tác làm dữ liệu

Dữ liệu là PTB-XL @wagner, công khai trên PhysioNet (truy cập #link("https://physionet.org/content/ptb-xl/1.0.3/")[tại đây]). Khóa luận sinh dữ liệu ba tác vụ từ chú giải SCP, áp quy tắc gán nhãn hai tầng (Mục 3.4), và cân bằng tập huấn luyện bằng cách giới hạn lớp đa số ở 1500 mẫu mỗi tác vụ, thu được khoảng 16 nghìn mẫu. Split tuân theo chuẩn Strodthoff: fold 1–8 huấn luyện, fold 9 kiểm định, fold 10 kiểm thử @strodthoff.

=== Chiến lược huấn luyện

Tinh chỉnh QLoRA 4-bit (r = 16, alpha = 32, lr 2e-4, fp16, sdpa) trên mô hình ngôn ngữ trong 2 epoch, đông cứng nhánh thị giác. Lõi PULSE không bị sửa đổi.

=== Độ đo sử dụng

Chỉ số chính là macro-AUC (không phụ thuộc ngưỡng), báo cáo riêng cho từng tác vụ. macro-F1 được báo cáo tham khảo, với ngưỡng per-class hiệu chuẩn trên tập kiểm định fold-9 rồi đóng băng khi đánh giá tập kiểm thử fold-10 (tránh chọn ngưỡng trên chính tập kiểm thử). Ngoài ra theo dõi xác suất trung bình P(1) trên lớp dương/âm. Ở phân tích lâm sàng (Mục 4.3) bổ sung các chỉ số chuẩn của một xét nghiệm chẩn đoán: độ nhạy, độ đặc hiệu, PPV, NPV cùng số ca dương tính giả/âm tính giả.

=== Khả năng tái lập

Toàn bộ thực nghiệm được thiết kế để tái lập được từ dữ liệu công khai đến con số cuối:

- *Dữ liệu*: PTB-XL phiên bản 1.0.3, công khai trên PhysioNet (truy cập #link("https://physionet.org/content/ptb-xl/1.0.3/")[tại đây]); nhãn đọc trực tiếp từ tệp `scp_statements.csv` chính thức, không chép tay danh sách mã.
- *Phân chia tập*: dùng nguyên cột fold khuyến nghị của Strodthoff @strodthoff (fold 1–8 huấn luyện, fold 9 kiểm định, fold 10 kiểm thử), không tự ý chia ngẫu nhiên — bảo đảm so sánh đồng nhất với các công trình khác trên cùng split.
- *Quy tắc gán nhãn xác định*: hai tầng (chẩn đoán theo ngưỡng likelihood ≥ 50%, nhịp/hình thái theo hiện diện) sinh tự động từ `scp_codes`, không có lựa chọn thủ công (Mục 3.4).
- *Mô hình và siêu tham số*: nền PULSE-7B công khai trên Hugging Face; QLoRA 4-bit r = 16, alpha = 32, dropout 0.05, lr 2e-4, fp16, attention sdpa, 2 epoch, đông cứng nhánh thị giác, không sửa lõi (Mục 3.5). Cấu hình anyres lấy đúng từ `config.json` của PULSE-7B nên xử lý ảnh giống hệt giữa mô hình nền và mô hình tinh chỉnh.
- *Đánh giá xác định*: giao thức teacher-forced chỉ một forward pass, KHÔNG lấy mẫu, không nhiệt độ ngẫu nhiên — cùng đầu vào luôn cho cùng điểm số, không phụ thuộc seed sinh; ngưỡng macro-F1 đóng băng từ fold-9; macro-AUC không phụ thuộc ngưỡng.
- *Tổ chức mã nguồn và mức tái sử dụng PULSE*: toàn bộ pipeline gồm hai khối tách bạch. Khối _dùng lại từ PULSE_ là fork LLaVA của PULSE (thư mục `source/LLaVA`, toàn bộ stack huấn luyện và suy luận) — khóa luận KHÔNG sửa lõi: bước huấn luyện gọi thẳng hàm `llava.train.train` với mục tiêu entropy chéo gốc, bước suy luận nạp mô hình qua chính các mô-đun `llava`. Khối _mã của khóa luận_ là một lớp mỏng gom trong một thư mục duy nhất (`khoaluan/code`): sinh dữ liệu, quy tắc nhãn, prompt, launcher huấn luyện, đánh giá teacher-forced (`eval_v2.py`) và gộp Super (`aggregate_super5.py`). Nghĩa là đóng góp kỹ thuật nằm ở lớp dữ liệu–đánh giá, còn năng lực nền được kế thừa gần như nguyên vẹn từ PULSE — điều này cũng giải thích vì sao mô hình nền PULSE-7B đã đạt macro-AUC cao ngay khi chưa tinh chỉnh (Mục 4.3). Các script cho phép tái dựng mọi bảng kết quả từ checkpoint và tệp dump; phép gộp 44 mã → 5 siêu lớp là hậu xử lý xác định theo cột `diagnostic_class`.

Hạn chế còn lại về tái lập: ngưỡng ở phần phân tích Super/lâm sàng (Bảng 4.5, 4.6) hiện dò trên chính tập kiểm thử (lạc quan); để hoàn toàn không rò rỉ cần một lần đánh giá fold-9 rồi đóng băng ngưỡng — đã chuẩn bị sẵn quy trình nhưng chưa chạy do giới hạn tài nguyên. macro-AUC không bị ảnh hưởng bởi hạn chế này.

== Kết quả thực nghiệm

=== Kết quả theo từng tác vụ

Bảng 4.1 trình bày kết quả của mô hình cuối (tinh chỉnh 2 epoch) trên tập kiểm thử độc lập fold-10 — số headline không thiên lệch vì fold-10 không tham gia huấn luyện. Ở cả ba tác vụ, xác suất trung bình P(1) của lớp dương lớn hơn hẳn lớp âm (chẩn đoán 0.44 so với 0.03; nhịp 0.64 so với 0.06; hình thái 0.35 so với 0.05), cho thấy mô hình thực sự phân biệt được nhãn có và không, không rơi vào sụp đổ chế độ. macro-AUC được lấy trung bình trên các lớp có ít nhất một mẫu dương trong fold-10 (chẩn đoán 44/44, nhịp 11/12, hình thái 19/19 — lớp nhịp còn lại không xuất hiện mẫu dương nên AUC không xác định và bị loại khỏi trung bình, xem cột "Số lớp có AUC"). Những lớp tốt nhất gồm clbbb 1.00, injin/injil 1.00 và crbbb/3avb 0.99 (chẩn đoán), stach và afib 0.99 (nhịp), pvc 0.99 và prc(s) 0.98 (hình thái).

#figure(
  table(
    columns: 6,
    inset: 6pt,
    align: (left, center, center, center, center, center),
    table.header([*Tác vụ*], [*Số lớp có AUC*], [*macro-AUC*], [*macro-F1*], [*mean P(1) dương*], [*mean P(1) âm*]),
    [Chẩn đoán (44)], [44/44], [0.893], [0.24], [0.44], [0.03],
    [Nhịp (12)], [11/12], [0.887], [0.33], [0.64], [0.06],
    [Hình thái (19)], [19/19], [0.849], [0.22], [0.35], [0.05],
    [Trung bình], [–], [0.876], [0.26], [–], [–],
  ),
  caption: [Bảng 4.1: Kết quả phân lớp theo từng tác vụ trên PTB-XL (fold-9 kiểm định và fold-10 kiểm thử).],
  kind: table,
)

Lưu ý về giao thức ngưỡng: macro-F1 được tính với ngưỡng per-class hiệu chuẩn trên tập kiểm định fold-9 rồi đóng băng khi đánh giá trên tập kiểm thử fold-10, nhằm tránh rò rỉ thông tin từ tập kiểm thử; do đó các giá trị F1 ở đây là ước lượng trung thực (thấp hơn so với khi dò ngưỡng trực tiếp trên tập kiểm thử). Ngược lại, macro-AUC không phụ thuộc ngưỡng nên không bị ảnh hưởng bởi lựa chọn này.

Việc huấn luyện thêm cải thiện đều trên cả ba tác vụ và chưa có dấu hiệu khớp quá: macro-AUC trung bình trên fold-10 tăng từ 0.818 sau epoch 1 lên 0.876 sau epoch 2 (chẩn đoán 0.852 → 0.893, nhịp 0.833 → 0.887, hình thái 0.768 → 0.849). Do đó các con số báo cáo nên được hiểu là cận dưới, còn dư địa cải thiện nếu huấn luyện thêm.

=== So sánh với các mốc tham chiếu

Trên fold-10, trung bình macro-AUC 0.876 cao hơn rõ rệt các backbone tổng quát chưa tinh chỉnh ECG vốn ở mức gần ngẫu nhiên (GPT-4o 0.556, LLaVA-1.6 0.500) @pulse — các mốc này đo trên phân nhóm Super 5 lớp nên cũng không phải so sánh tương đương. So với các hệ chuyên biệt báo cáo trên phân nhóm Super 5 lớp (PULSE 0.824 @pulse, GEM 0.834 @gem), về trị tuyệt đối con số của khóa luận cao hơn; tuy nhiên cần thận trọng khi đối chiếu vì hai mốc Super chỉ phân loại 5 siêu lớp, trong khi khóa luận đánh giá trên toàn bộ tập nhãn chi tiết (44+12+19 mã) khó hơn nhiều — do đó đây không phải so sánh tương đương. So sánh tương đương duy nhất là với chính mô hình nền PULSE-7B trên cùng benchmark (Mục 4.3). So với mức trần đơn phương thức của Strodthoff (chẩn đoán 0.937, nhịp 0.957, hình thái 0.896) @strodthoff vẫn còn khoảng cách (chênh khoảng 0.04–0.07), hợp lý vì baseline làm việc trực tiếp trên tín hiệu một chiều đầy đủ còn khóa luận chỉ dùng ảnh ECG.

#figure(
  table(
    columns: 3,
    inset: 6pt,
    align: (left, center, center),
    table.header([*Hệ thống*], [*Phương thức*], [*macro-AUC*]),
    [Thử nghiệm ban đầu (loss phân loại đa nhãn)], [ảnh + text], [~0.5 (sụp về toàn NORM)],
    [Mô hình đề xuất (test fold-10)], [ảnh + text], [0.876 (TB); chẩn đoán 0.893 / nhịp 0.887 / hình thái 0.849],
    [PULSE-7B base zero-shot (test fold-10, cùng eval)], [ảnh + text], [0.779 (TB); 0.835 / 0.774 / 0.728],
    [Strodthoff CNN 1-D (trần) @strodthoff], [tín hiệu], [0.937 / 0.957 / 0.896],
    [PULSE-7B zero-shot (Super, 5 lớp) @pulse], [ảnh + text], [0.824],
    [GEM (Super) @gem], [đa phương thức], [0.834],
    [GPT-4o zero-shot (Super) @pulse], [ảnh + text], [0.556],
    [LLaVA-1.6 backbone zero-shot (Super) @pulse], [ảnh + text], [0.500],
  ),
  caption: [Bảng 4.2: So sánh với các mốc tham chiếu (macro-AUC).],
  kind: table,
)

== So sánh trực tiếp với PULSE gốc

Bài báo gốc PULSE chỉ công bố macro-AUC trên PTB-XL Super gồm 5 siêu lớp, không báo cáo trên tập nhãn chi tiết 44/12/19 @pulse. Để có một so sánh thực sự công bằng với mô hình nền, khóa luận chạy chính PULSE-7B gốc (chưa qua QLoRA, không gắn adapter) qua đúng giao thức teacher-forced trên cùng tập kiểm thử fold-10 và cùng ba tập nhãn. Kết quả ở Bảng 4.3.

#figure(
  table(
    columns: 4,
    inset: 6pt,
    align: (left, center, center, center),
    table.header([*Chỉ số / Tác vụ*], [*PULSE-7B base (zero-shot)*], [*Mô hình đề xuất*], [*Δ*]),
    table.cell(colspan: 4, align: left)[_macro-AUC (chỉ số chính, cùng cỡ mẫu — toàn fold-10)_],
    [Chẩn đoán (44)], [0.835], [0.893], [+0.058],
    [Nhịp (12)], [0.774], [0.887], [+0.113],
    [Hình thái (19)], [0.728], [0.849], [+0.121],
    [Trung bình], [0.779], [0.876], [*+0.097*],
    table.cell(colspan: 4, align: left)[_macro-F1 (val-frozen) — †so sánh định hướng, khác cỡ mẫu_],
    [Chẩn đoán (44)], [0.17], [0.24], [+0.07],
    [Nhịp (12)], [0.27], [0.33], [+0.06],
    [Hình thái (19)], [0.15], [0.22], [+0.07],
    [Trung bình], [≈0.19], [0.26], [≈+0.07],
  ),
  caption: [Bảng 4.3: So sánh trực tiếp PULSE-7B gốc với mô hình đề xuất (cùng fold-10, cùng giao thức).],
  kind: table,
)
#text(size: 11pt)[† macro-F1 của PULSE-base ước lượng trên 150 mẫu mỗi tác vụ (do chi phí suy luận), còn macro-F1 của mô hình đề xuất đo trên toàn bộ fold-10; vì khác cỡ mẫu nên cột Δ của macro-F1 chỉ mang tính định hướng, không phải so sánh chặt chẽ như macro-AUC.]

Hai quan sát đáng chú ý. Thứ nhất, PULSE-base đã đạt trung bình 0.779 dù chưa hề được huấn luyện trên định dạng nhãn nhị phân của khóa luận, cho thấy biểu diễn thị giác ECG học sẵn đã mang thông tin phân biệt mạnh và giao thức teacher-forced khai thác được thông tin đó. Thứ hai, việc tinh chỉnh mang lại mức tăng nhất quán và đáng kể trên cả ba tác vụ (trung bình +0.097), lớn nhất ở hình thái (+0.121) và nhịp (+0.113) — hai nhóm mã mà thử nghiệm ban đầu đánh mất hoàn toàn — và nhỏ hơn ở chẩn đoán (+0.058) vốn đã được PULSE đặt trọng tâm. Đây là so sánh tương đương duy nhất với mô hình nền (cùng tập kiểm thử, cùng tập nhãn, cùng giao thức), nên +0.097 phản ánh thuần đóng góp của bước tinh chỉnh trong khóa luận. Bảng 4.3 cũng bổ sung phần macro-F1 (val-frozen) cho cả hai mô hình: PULSE-base đạt trung bình khoảng 0.19 so với 0.26 của mô hình đề xuất, tăng nhất quán trên cả ba tác vụ (khoảng +0.06 đến +0.07). Điều này cho thấy bước tinh chỉnh cải thiện đồng thời cả macro-AUC lẫn macro-F1. Tuy vậy cần thận trọng với phần F1: hai giá trị được đo trên cỡ mẫu khác nhau (base ước lượng trên 150 mẫu mỗi tác vụ, mô hình đề xuất trên toàn bộ fold-10), nên đây chỉ là so sánh mang tính định hướng chứ chưa phải một khẳng định chặt chẽ. macro-AUC vẫn là chỉ số chính do F1 nhạy với việc chọn ngưỡng và phân phối lớp hiếm.

=== Đối chiếu trực tiếp trên cùng phân nhóm Super 5 lớp của PULSE

Bảng 4.2 đã lưu ý rằng việc đặt con số toàn-nhãn-chi-tiết của khóa luận (44+12+19 mã) cạnh con số Super 5 lớp mà PULSE công bố là _không tương đương_, vì hai bài toán khác độ khó. Để xoá bỏ hẳn sự khập khiễng này và tạo một đối chiếu thực sự cùng thước đo với bài báo gốc, khóa luận đưa chính mô hình của mình _về đúng phân nhóm Super 5 lớp_ mà PULSE đánh giá: gộp 44 mã chẩn đoán thành 5 siêu lớp NORM/MI/STTC/CD/HYP theo cột `diagnostic_class` của `scp_statements.csv` (xác suất siêu lớp = max xác suất các mã con, nhãn siêu lớp = hợp của các mã con — đúng định nghĩa siêu lớp của PTB-XL @wagner @strodthoff). Phép gộp này chỉ là hậu xử lý đầu ra, không cần huấn luyện lại, và được áp _y hệt_ cho cả PULSE-7B gốc lẫn mô hình đề xuất trên toàn bộ fold-10 (N = 2050). Kết quả ở Bảng 4.4.

#figure(
  table(
    columns: 5,
    inset: 6pt,
    align: (left, center, center, center, center),
    table.header([*Hệ thống (PTB-XL Super, 5 lớp)*], [*macro-AUC*], [*macro-F1*], [*Hamming*], [*Cỡ mẫu*]),
    [PULSE-7B — số công bố trong bài báo @pulse], [0.824], [0.748], [11.0%], [test PTB-XL],
    [PULSE-7B base — pipeline của khóa luận], [0.826], [0.615†], [20.9%†], [fold-10 (2050)],
    [Mô hình đề xuất — pipeline của khóa luận], [*0.902*], [0.693†], [13.6%†], [fold-10 (2050)],
  ),
  caption: [Bảng 4.4: Đối chiếu trên cùng phân nhóm Super 5 lớp với PULSE (gộp 44 mã chẩn đoán → 5 siêu lớp, cùng fold-10).],
  kind: table,
)
#text(size: 11pt)[† macro-F1 và Hamming ở hai hàng dưới được tính với phép gộp max và ngưỡng dò tối đa F1 từng siêu lớp ngay trên tập test (lạc quan, chỉ để tham khảo); chúng KHÔNG dùng cùng quy trình tạo nhãn/chọn ngưỡng với con số 0.748 / 11.0% của PULSE nên không phải so sánh chặt. Chỉ macro-AUC (không phụ thuộc ngưỡng) là đối chiếu sạch.]

Bảng 4.4 cho hai kết luận mạnh. Thứ nhất, *pipeline của khóa luận là trung thực và đáng tin*: chạy đúng PULSE-7B gốc qua pipeline của khóa luận rồi gộp về Super cho macro-AUC 0.826 — gần như trùng khít con số 0.824 mà bài báo PULSE công bố. Vì cùng một mô hình nền, sự khớp này chứng tỏ toàn bộ chuỗi xử lý của khóa luận (định dạng ảnh, anyres, giao thức teacher-forced) tái lập đúng năng lực mà PULSE báo cáo, không hề thổi phồng. Thứ hai, *trên chính benchmark của PULSE, mô hình đề xuất vượt mô hình gốc*: macro-AUC tăng từ 0.824/0.826 lên *0.902* (+0.078 so với số công bố), nhất quán với mức +0.097 đo trên tập nhãn chi tiết ở Bảng 4.3.

Phần macro-F1 ở Bảng 4.4 còn củng cố lập luận tại Mục 4.4.2 về việc vì sao phải lấy macro-AUC làm chỉ số chính: cùng là PULSE-base, pipeline của khóa luận tái lập đúng macro-AUC của bài báo (0.826 ≈ 0.824) nhưng lại cho macro-F1 thấp hơn hẳn (0.615 so với 0.748). Vì AUC trùng mà F1 lệch trên _cùng một mô hình_, chênh lệch F1 này thuần là hệ quả của khác biệt quy trình tạo nhãn và chọn ngưỡng (gộp max, ngưỡng tự dò), KHÔNG phải khác biệt năng lực — đúng như đã phân tích, F1 nhạy với hiệu chuẩn/ngưỡng còn macro-AUC thì không. Đây chính là lý do mọi kết luận headline của khóa luận đặt trên macro-AUC.

Bảng 4.5 tách macro-F1 trên Super thành từng siêu lớp, đối chiếu PULSE-base với mô hình đề xuất dưới _cùng_ phép gộp và cùng cách chọn ngưỡng (nên so sánh base-với-ta là công bằng, dù trị tuyệt đối còn lạc quan như đã nêu). Tinh chỉnh nâng F1 ở bốn trên năm siêu lớp, mạnh nhất ở STTC (+0.199) — nhóm bất thường tái cực/ST-T mà mô hình nền yếu nhất; riêng HYP gần như đứng yên (−0.010), phù hợp với việc đây là siêu lớp ít mẫu dương nhất (222) và khó nhất.

#figure(
  table(
    columns: 4,
    inset: 6pt,
    align: (left, center, center, center),
    table.header([*Siêu lớp (số mẫu dương)*], [*PULSE-base*], [*Mô hình đề xuất*], [*Δ*]),
    [NORM (954)], [0.793], [0.854], [+0.061],
    [MI (415)], [0.588], [0.682], [+0.094],
    [STTC (506)], [0.509], [0.708], [+0.199],
    [CD (496)], [0.665], [0.710], [+0.045],
    [HYP (222)], [0.522], [0.512], [−0.010],
    [*macro-F1*], [0.615], [*0.693*], [*+0.078*],
  ),
  caption: [Bảng 4.5: macro-F1 theo từng siêu lớp PTB-XL Super (fold-10, N = 2050; ngưỡng self-tuned cùng cách cho cả hai mô hình).],
  kind: table,
)
#text(size: 11pt)[Các giá trị F1 dùng ngưỡng dò tối đa F1 từng lớp ngay trên tập test (lạc quan); chúng KHÔNG so trực tiếp được với F1 74.8% của PULSE (khác quy trình ngưỡng) mà chỉ để đối chiếu PULSE-base với mô hình đề xuất dưới cùng một thước đo. macro-AUC tương ứng: PULSE-base 0.826, mô hình đề xuất 0.902.]

=== Phân tích lâm sàng: giá trị sai/sót và độ nhạy–độ đặc hiệu

macro-AUC và macro-F1 chưa nói trực tiếp điều mà bác sĩ quan tâm nhất khi đọc một mô hình hỗ trợ chẩn đoán: mô hình _bỏ sót_ bao nhiêu ca bệnh (âm tính giả) và _báo nhầm_ bao nhiêu ca (dương tính giả), vì trong y khoa hai loại lỗi này có chi phí rất khác nhau — bỏ sót một ca nhồi máu cơ tim nguy hiểm hơn nhiều so với một báo động giả. Do đó khóa luận phân rã kết quả Super của mô hình đề xuất thành ma trận nhầm lẫn từng siêu lớp và quy ra các chỉ số chuẩn của một xét nghiệm chẩn đoán: độ nhạy (sensitivity = tỷ lệ ca bệnh được bắt đúng, càng cao càng ít bỏ sót), độ đặc hiệu (specificity = tỷ lệ ca lành được loại đúng, càng cao càng ít báo nhầm), giá trị tiên đoán dương PPV và âm NPV (Bảng 4.6, cùng điểm vận hành ngưỡng với Bảng 4.5).

#figure(
  table(
    columns: 7,
    inset: 5pt,
    align: (left, center, center, center, center, center, center),
    table.header([*Siêu lớp*], [*Dương tính giả (sai)*], [*Âm tính giả (sót)*], [*Độ nhạy*], [*Độ đặc hiệu*], [*PPV*], [*NPV*]),
    [NORM], [172], [115], [0.879], [0.843], [0.830], [0.889],
    [MI], [168], [113], [0.728], [0.897], [0.643], [0.928],
    [STTC], [259], [87], [0.828], [0.832], [0.618], [0.937],
    [CD], [163], [133], [0.732], [0.895], [0.690], [0.913],
    [HYP], [51], [128], [0.423], [0.972], [0.648], [0.933],
    [*macro*], [–], [–], [*0.718*], [*0.888*], [*0.686*], [*0.920*],
  ),
  caption: [Bảng 4.6: Phân tích sai/sót và chỉ số chẩn đoán của mô hình đề xuất trên PTB-XL Super (fold-10, N = 2050).],
  kind: table,
)

Ba nhận định lâm sàng. Thứ nhất, *độ đặc hiệu và NPV cao* (macro 0.888 và 0.920): khi mô hình báo một siêu lớp là âm tính thì phần lớn là đúng, nên mô hình phù hợp vai trò sàng lọc _loại trừ_ — một ECG được gắn "không bất thường" ở một nhóm có độ tin cậy cao. Thứ hai, *tinh chỉnh chủ yếu cắt giảm dương tính giả*: so với PULSE-base (độ đặc hiệu macro 0.795), mô hình đề xuất nâng độ đặc hiệu lên 0.888 và giảm mạnh số báo nhầm — rõ nhất ở STTC, số dương tính giả rớt từ 732 (base) xuống 259 — trong khi vẫn giữ độ nhạy tương đương hoặc cao hơn, tức ít làm phiền bác sĩ bằng cảnh báo thừa mà không đánh đổi việc bỏ sót. Thứ ba, *điểm yếu phải nêu thẳng là độ nhạy ở nhóm phì đại HYP chỉ 0.423* — mô hình bỏ sót 128 trên 222 ca, nên ở nhóm này nó CHƯA đủ tin cậy để loại trừ bệnh; nhóm MI và CD cũng còn bỏ sót đáng kể (độ nhạy 0.73). Đây là giới hạn cần ghi rõ trước bất kỳ ý định dùng lâm sàng.

Một điểm quan trọng về _điểm vận hành_: các con số trên lấy tại ngưỡng tối đa F1, không phải ngưỡng tối ưu cho lâm sàng. Vì macro-AUC cao (0.902), đường ROC cho phép dịch ngưỡng sang phía _ưu tiên độ nhạy_ để giảm bỏ sót ở những nhóm chi phí âm-tính-giả cao như MI, đánh đổi bằng độ đặc hiệu thấp hơn; AUC cao chính là bảo đảm rằng đánh đổi này thuận lợi. Việc chọn ngưỡng theo bất đối xứng chi phí lâm sàng (ví dụ cố định độ nhạy ≥ 0.95 cho MI) là một tham số triển khai, được để ngỏ như hướng phát triển và cần hiệu chuẩn trên tập kiểm định độc lập thay vì dò trên tập kiểm thử như ở đây.

== Nghiên cứu cắt bỏ và phân tích

=== Thử nghiệm ban đầu làm cơ sở chuyển hướng

Một thử nghiệm ban đầu coi toàn bộ bài toán chẩn đoán là một bài phân loại đa nhãn duy nhất: mô hình sinh một khối 44 nhãn, mỗi mã một token "1"/"0", và được giám sát bằng một hàm mất mát kiểu phân loại đa nhãn áp lên xác suất token. Thử nghiệm này thất bại trên hai trục: sụp đổ về dự đoán gần như toàn NORM (macro-AUC khoảng 0.5) và khoảng 73% mẫu cho khối nhãn hỏng khi sinh tự do. Khóa luận suy đoán nguyên nhân là sự không tương thích giữa một hàm mất mát kiểu phân loại (giả định nhãn độc lập) và một decoder sinh token, cộng với mất cân bằng lớp nặng @welleck, @ptbxlimb. Thất bại này là động lực để chuyển sang ba thiết kế của mô hình đề xuất: phân rã theo tác vụ, trở về entropy chéo gốc và đánh giá teacher-forced.

=== Phân tích chỉ số và hiện tượng khớp quá

Khoảng cách giữa macro-AUC cao (0.876) và macro-F1 thấp (0.26) bắt nguồn từ việc hai chỉ số đo hai thứ khác nhau, không phải từ năng lực ra quyết định yếu của mô hình: macro-AUC đo khả năng _xếp hạng_ — mô hình có gán điểm xác suất cao hơn cho mẫu dương so với mẫu âm hay không — và không phụ thuộc vào việc chọn ngưỡng; trong khi macro-F1 đo chất lượng _quyết định nhị phân_ tại MỘT ngưỡng cụ thể và bị chi phối mạnh bởi việc chọn ngưỡng đó. Một mô hình hoàn toàn có thể xếp hạng tốt (AUC cao) nhưng F1 thấp nếu ngưỡng vận hành chưa tối ưu cho từng lớp — đây là vấn đề _hiệu chuẩn/chọn ngưỡng_, không phải thất bại về khả năng phân biệt. Cụ thể, với ngưỡng per-class hiệu chuẩn trên fold-9 và đóng băng khi đánh giá fold-10, macro-F1 trên fold-10 còn thấp (khoảng 0.22–0.33; chẩn đoán 0.24, nhịp 0.33, hình thái 0.22) vì hai lý do: (i) phép macro-trung bình tính đều cả các lớp cực hiếm của phân phối dài đuôi PTB-XL, nơi chỉ vài mẫu dương nên một vài dự đoán sai đã kéo F1 của lớp đó xuống gần 0; và (ii) ngưỡng học trên val không khớp hoàn hảo với test ở chính các lớp hiếm này. Vì vậy macro-AUC (không phụ thuộc ngưỡng) được chọn làm chỉ số chính, và việc nâng F1 thuộc về bước hiệu chuẩn ngưỡng — chẳng hạn tối ưu ngưỡng theo F1 cho từng lớp hoặc báo cáo bổ sung micro-F1 (ít bị lớp hiếm chi phối) — được để ngỏ như hướng phát triển. Trong ba tác vụ, hình thái yếu nhất (0.849), với lớp tab chỉ đạt khoảng 0.34 — phù hợp với độ khó nội tại của các phát biểu hình thái và mức trần thấp hơn của tác vụ này @strodthoff. Vì hiệu năng còn tăng từ epoch 1 sang epoch 2, mô hình chưa bão hòa và việc dừng ở 2 epoch là do giới hạn tài nguyên chứ không phải do hội tụ; đây vừa là hạn chế vừa là dư địa cải thiện.

=== Nghiên cứu cắt bỏ: mô hình có thực sự dùng ảnh ECG và có rò rỉ nhãn không?

Để trả lời hai chất vấn quan trọng — (i) mô hình có thực sự khai thác ảnh ECG hay chỉ dựa vào thiên kiến nhãn, và (ii) giao thức teacher-forced có rò rỉ nhãn (label leakage) hay không — khóa luận chạy bốn biến thể đánh giá trên cùng mô hình cuối (checkpoint epoch-2), cùng tập kiểm thử fold-10 (lấy 80 mẫu mỗi tác vụ để cân đối chi phí), chỉ thay đổi đầu vào ảnh hoặc token điền:

- *real*: ảnh ECG đúng (điều kiện tham chiếu).
- *blank*: thay ảnh bằng một ảnh xám đồng nhất (xoá toàn bộ tín hiệu thị giác).
- *shuffle*: gán cho mỗi mẫu ảnh của một mẫu khác (phá vỡ tương ứng ảnh–nhãn).
- *placeholder=1*: giữ ảnh đúng nhưng đổi token điền ở mọi ô nhãn từ "0" thành "1".

#figure(
  table(
    columns: 5,
    inset: 6pt,
    align: (left, center, center, center, center),
    table.header([*Điều kiện*], [*Chẩn đoán*], [*Nhịp*], [*Hình thái*], [*Trung bình*]),
    [real (ảnh đúng)], [0.936], [0.968], [0.931], [0.945],
    [blank (ảnh xám)], [0.500], [0.500], [0.500], [0.500],
    [shuffle (ảnh sai)], [0.454], [0.366], [0.461], [0.427],
    [placeholder = "1"], [0.833], [0.722], [0.839], [0.798],
  ),
  caption: [Bảng 4.7: Nghiên cứu cắt bỏ — ảnh đúng/xám/đảo và kiểm soát token điền (macro-AUC, fold-10).],
  kind: table,
)

Hai kết luận. Thứ nhất, *mô hình thực sự dùng ảnh ECG*: khi xoá ảnh (blank) hoặc gán sai ảnh (shuffle), macro-AUC sụp từ 0.945 về mức ngẫu nhiên (0.500 và 0.427) ở cả ba tác vụ. Vì khối nhãn answer giữ nguyên hệt nhau giữa các điều kiện và chỉ ảnh thay đổi, kết quả này đồng thời *bác bỏ nguy cơ rò rỉ nhãn*: nếu điểm số được rút từ nhãn nằm trong khối answer thì blank/shuffle vẫn phải cho AUC cao — đằng này AUC sụp về ngẫu nhiên, chứng tỏ toàn bộ tín hiệu phân biệt đến từ ảnh chứ không từ khối nhãn. Thứ hai, biến thể *placeholder = "1"* vẫn cho AUC cao hơn hẳn mức ngẫu nhiên (trung bình 0.798 so với 0.500), xác nhận điểm số là phân phối mà mô hình _dự đoán_ tại ô nhãn chứ không phải sao chép token đã điền sẵn; mức giảm so với real (0.945 → 0.798) phản ánh việc một tiền tố toàn "1" là ngữ cảnh tự hồi quy bất thường làm lệch hiệu chuẩn, chứ không phải rò rỉ nhãn thật (nhãn thật của lớp đang chấm không bao giờ được đưa vào chuỗi đầu vào dưới bất kỳ token điền nào).

Biến thể placeholder = "1" là một _phép thử áp lực ngoài phân phối_ (out-of-distribution) được cố ý dựng ra chỉ để kiểm tra giả thuyết rò rỉ nhãn, KHÔNG phải điều kiện vận hành. Giao thức đánh giá thật luôn dùng cùng một token điền "0" một cách nhất quán cho cả mô hình nền lẫn mô hình tinh chỉnh và cho mọi lớp, nên mọi số liệu headline (Bảng 4.1–4.6) đều đo dưới cùng một điều kiện cố định và so sánh giữa chúng là công bằng. Việc điểm số dịch chuyển khi ép một tiền tố toàn "1" là đặc tính chung của _mọi_ mô hình ngôn ngữ tự hồi quy khi bị đặt vào ngữ cảnh bất thường, và đã được kiểm soát bằng cách giữ token điền bất biến; nó nói lên rằng giá trị tuyệt đối của xác suất chưa được hiệu chuẩn hoàn hảo (nhất quán với phần thảo luận macro-F1 ở Mục 4.4.2), chứ không làm suy giảm độ tin cậy của các kết luận dựa trên macro-AUC vốn chỉ phụ thuộc thứ tự xếp hạng.
