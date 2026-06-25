= Phương pháp nghiên cứu

== Tổng quan phương pháp

Phương pháp giữ nguyên lõi đa phương thức của PULSE-7B và chỉ can thiệp ở bốn tầng: định dạng bài toán, gán nhãn, hàm mục tiêu và giao thức đánh giá. Ảnh ECG được đưa qua bộ mã hóa thị giác, các đặc trưng thị giác được chiếu sang không gian ngôn ngữ và nối với câu hỏi tác vụ để mô hình ngôn ngữ sinh khối nhãn dạng văn bản.

#figure(
  image("../Hinh_3_1_detail.png", width: 100%),
  caption: [Hình 3.1: Tổng quan kiến trúc mô hình đề xuất (PULSE-7B / LLaVA-v1.6-Vicuna-7B). Ảnh ECG được tile theo anyres rồi mã hóa bằng CLIP đông cứng và chiếu sang không gian ngôn ngữ; các token thị giác được chèn vào chuỗi văn bản (fusion), đi qua mô hình ngôn ngữ tinh chỉnh QLoRA, rồi đọc xác suất theo giao thức teacher-forced; bảng "Details" mô tả nội khối từng thành phần. Ba ý niệm cốt lõi: (i) _anyres_ — với `image_grid_pinpoints` (tối đa 1008 px), ảnh 1344×672 bị thu nhỏ về 672×336 rồi cắt 2×1 cộng một ảnh toàn cục thành 3 ô, cho 3×576 = 1728 token thị giác; (ii) _fusion_ — không có cross-attention riêng, token thị giác (sau khi projector đưa về chiều 4096) được ghép thẳng vào chuỗi văn bản tại vị trí ⟨image⟩ rồi cả hai cùng đi qua một stack self-attention nhân quả; (iii) _QLoRA_ — trọng số nền Vicuna-7B lượng tử hóa 4-bit (NF4) và đông cứng, chỉ học cặp hạng thấp $A, B$ (r=16) ở mỗi lớp tuyến tính, còn bộ mã hóa thị giác, projector và LM head đều đông cứng.],
  kind: image,
)

== Kiến trúc đa phương thức và chiến lược tinh chỉnh

=== Bộ mã hóa thị giác và lớp chiếu

Mô hình kế thừa CLIP ViT-L/14 ở độ phân giải 336 và lớp chiếu mlp2x_gelu của PULSE @pulse. Mỗi ô ảnh 336×336 được CLIP chia thành lưới 24×24 = 576 token (bỏ token CLS, lấy đặc trưng lớp áp chót), rồi lớp chiếu mlp2x_gelu ánh xạ mỗi token từ 1024 chiều sang 4096 chiều của mô hình ngôn ngữ.

Ảnh ECG được xử lý theo cơ chế anyres (giữ nguyên cấu hình của PULSE) để hỗ trợ ảnh nhiều kích thước. Cần mô tả chính xác cơ chế này như nó thực sự chạy trong khóa luận: ảnh điện tâm đồ kết xuất ở kích thước 1344×672. Hàm `process_anyres_image` chọn độ phân giải khớp nhất trong tập `image_grid_pinpoints` của mô hình (giá trị lớn nhất là 1008), nên ảnh trước hết bị thu nhỏ về 672×336; sau đó được cắt thành lưới 2×1 (hai ô cục bộ 336×336, ứng với nửa trái — các chuyển đạo chi, và nửa phải — các chuyển đạo trước tim), kèm một ảnh thu nhỏ toàn cục (toàn bộ ảnh ép về 336×336). Với `mm_patch_merge_type = "flat"`, tổng cộng có 3 ô × 576 = 1728 token thị giác đưa vào mô hình ngôn ngữ. Như vậy anyres ở cấu hình hiện tại chủ yếu chia đôi ảnh theo chiều ngang chứ không bảo toàn được độ phân giải gốc 1344×672; đây là một giới hạn được thảo luận ở Mục 5.2. Cơ chế tiling được minh họa ở Hình 3.2.

#figure(
  image("../Hinh_3_3_tiling.png", width: 100%),
  caption: [Hình 3.2: Cơ chế anyres thực sự áp dụng cho ảnh ECG 12 chuyển đạo. Ảnh 1344×672 bị thu nhỏ về 672×336, cắt thành lưới 2×1 (một đường cắt dọc → nửa trái là các chuyển đạo chi, nửa phải là các chuyển đạo trước tim) kèm một ảnh thu nhỏ toàn cục, tổng cộng 3 ô và 1728 token thị giác; mỗi ô vẫn đủ chiều cao (tất cả hàng + dải nhịp, bị ép dọc). Việc thu nhỏ về 672×336 làm mất chi tiết tinh của tín hiệu; tiling mịn hơn (2×2 hoặc 4×1 ≈ 2880 token) vẫn lọt cửa sổ 4096 và có thể giúp ích, còn lưới 4×2 đầy đủ (5184 token) cần mở rộng cửa sổ ngữ cảnh — để ngỏ ở Mục 5.2. Mã và cấu hình tiling giống hệt PULSE (cùng `process_anyres_image`, cùng `image_grid_pinpoints`), chỉ khác ảnh đầu vào; khác biệt định dạng so với PULSE nêu ở Bảng 3.1.],
  kind: image,
)

Sau bước chiếu, các token thị giác được hợp nhất với chuỗi văn bản theo cơ chế _early fusion_: 1728 token ảnh được chèn trực tiếp vào chuỗi token tại vị trí ⟨image⟩ rồi cùng đi qua bộ giải mã ngôn ngữ trong một stack self-attention nhân quả duy nhất, không qua một nhánh cross-attention riêng (Hình 3.3).

#figure(
  image("../Hinh_3_4_fusion.png", width: 100%),
  caption: [Hình 3.3: Cơ chế hợp nhất thị giác–ngôn ngữ (early fusion, joint self-attention). Sau khi projector đưa token ảnh từ 1024 về 4096 chiều (đúng chiều của mô hình ngôn ngữ), 1728 token ảnh được ghép thẳng vào chuỗi văn bản tại vị trí ⟨image⟩ — không có nhánh cross-attention riêng (khác Flamingo/BLIP-2). Cả hai luồng cùng đi qua một stack self-attention nhân quả: theo độ sâu là tuần tự (lớp 1 → … → 32), còn theo token thì ảnh và văn bản được xử lý đồng thời (văn bản co-attend với ảnh ở mọi lớp), không phải two-tower gộp muộn. Chiều ẩn giữ nguyên 4096 xuyên suốt; chỉ adapter LoRA (r=16) được học.],
  kind: image,
)

=== Tinh chỉnh QLoRA trên mô hình ngôn ngữ

Khóa luận đông cứng toàn bộ nhánh thị giác và lớp chiếu, chỉ tinh chỉnh mô hình ngôn ngữ bằng QLoRA 4-bit (NF4) với r = 16, alpha = 32, learning rate 2e-4, độ chính xác fp16 và cơ chế attention sdpa. Lõi PULSE không bị sửa đổi. Lựa chọn này vừa tiết kiệm tài nguyên tính toán, vừa bảo toàn năng lực biểu diễn ảnh ECG mà PULSE đã học sẵn. Mô hình được tinh chỉnh trong 2 epoch; do giới hạn tài nguyên tính toán, luận văn dừng ở 2 epoch dù đường cong hiệu năng vẫn còn đi lên (xem Mục 4.2).

== Tiền xử lý dữ liệu và định dạng instruction-tuning

*Từ tín hiệu PTB-XL đến ảnh và mẫu hội thoại.* PTB-XL cung cấp mỗi bản ghi ECG 12 chuyển đạo cùng trường `scp_codes` — một từ điển ánh xạ mỗi mã SCP sang giá trị likelihood (0–100). Quy trình tiền xử lý gồm bốn bước: (1) _kết xuất ảnh_ — mỗi tín hiệu ECG được vẽ thành ảnh 12 chuyển đạo và lưu dưới dạng PNG (`images/{ecg_id}.png`), vì PULSE nhận đầu vào là ẢNH ECG (qua CLIP) chứ không phải tín hiệu một chiều; (2) _đọc nhãn từ scp_codes_ — đọc trực tiếp từ điển likelihood để áp quy tắc gán nhãn hai tầng (Mục 3.4), nhờ đó giữ được cả ba nhóm chẩn đoán/nhịp/hình thái, khác với cách nhị phân hóa sớm của phiên bản trước vốn làm mất nhịp và hình thái; (3) _chia tập theo strat_fold_ — tuân thủ chuẩn Strodthoff (fold 1–8 huấn luyện, fold 9 kiểm định, fold 10 kiểm thử) để so sánh được với benchmark; (4) _sinh mẫu ba tác vụ_ — mỗi bản ghi ECG sinh ra ba mẫu (chẩn đoán, nhịp, hình thái) tái dùng chung một ảnh nhưng khác câu hỏi và khối nhãn.

*Định dạng ảnh ECG và đối chiếu với PULSE.* Ảnh được kết xuất bằng matplotlib ở kích thước 1344×672, theo bố cục 12 chuyển đạo chuẩn (3 hàng × 4 cột: hàng 1 I/aVR/V1/V4, hàng 2 II/aVL/V2/V5, hàng 3 III/aVF/V3/V6), mỗi ô in sẵn tên chuyển đạo, kèm một _dải nhịp_ chạy hết chiều ngang ở dưới cùng vẽ chuyển đạo II suốt 10 giây. Cần đối chiếu định dạng này với PULSE để thấy rõ điểm kế thừa và điểm khác biệt. Phần phụ lục của PULSE mô tả ảnh chuẩn là "12-lead ECG images ... black waveforms on a white background, red grid lines, and a 4x3 layout", sinh bằng công cụ ECG-image-kit và bổ sung nhiều biến dạng mô phỏng ảnh chụp thực (nếp nhăn, xoay, nhiễu, đổi màu nền, đổi độ phân giải, đôi khi bỏ lưới), cùng các bố cục thay thế 12×1 và 6×2 @pulse. Theo quy ước của PULSE ("6x2 (two rows of six leads)"), ký hiệu là (cột × hàng), nên "4x3" của PULSE chính là bố cục lưới chuẩn 4 cột × 3 hàng — _trùng với lưới của khóa luận_. Hai khác biệt thực sự (Bảng 3.1): (i) khóa luận render _sạch_ không biến dạng, trong khi PULSE cố ý thêm biến dạng; (ii) khóa luận _thêm dải nhịp_ chuyển đạo II 10 giây — toàn văn bài báo PULSE không hề nhắc tới "rhythm strip", nên đây là một bổ sung lệch khỏi phân phối huấn luyện của PULSE. Dải nhịp là quy ước lâm sàng giúp đánh giá nhịp trên đoạn dài; tuy nhiên đóng góp riêng của nó chưa được tách bằng nghiên cứu cắt bỏ nên khóa luận không khẳng định nó nâng hiệu năng, và để ngỏ như một hướng kiểm chứng (Mục 5.2).

#figure(
  table(
    columns: 3,
    inset: 6pt,
    align: (left, left, left),
    table.header([*Yếu tố*], [*Khóa luận*], [*PULSE @pulse*]),
    [Công cụ kết xuất], [matplotlib], [ECG-image-kit],
    [Bố cục 12 chuyển đạo], [lưới chuẩn 4×3 (3 hàng × 4 cột)], [lưới chuẩn 4×3 (+ 12×1, 6×2)],
    [Nhãn tên chuyển đạo], [có (in mỗi ô)], [ảnh 12-lead chuẩn (có nhãn)],
    [Dải nhịp (II, 10 giây)], [*có — bổ sung*], [*không nhắc tới trong bài báo*],
    [Biến dạng ảnh], [không (ảnh sạch)], [có: nhăn, xoay, nhiễu, đổi màu, đổi độ phân giải…],
    [Độ phân giải], [1344×672 cố định], [không nêu (chỉ "varying resolutions")],
  ),
  caption: [Bảng 3.1: Đối chiếu định dạng ảnh ECG giữa khóa luận và PULSE (nội dung PULSE trích từ phụ lục bài báo @pulse). Lưới 12 chuyển đạo là chung; khác biệt nằm ở dải nhịp (khóa luận thêm) và biến dạng (PULSE thêm).],
  kind: table,
)

*Định dạng instruction-tuning.* Tinh chỉnh theo chỉ dẫn (instruction tuning) là việc huấn luyện mô hình trên các cặp (chỉ dẫn, câu trả lời) để mô hình học cách làm theo yêu cầu của câu lệnh. PULSE bản thân đã được instruction-tuning trên tập ECGInstruct: nó học cách đọc một câu lệnh kèm ảnh ECG rồi sinh câu trả lời tương ứng @pulse. Khóa luận tận dụng đúng cơ chế đó — mỗi mẫu huấn luyện là một cặp (chỉ dẫn, câu trả lời) theo định dạng hội thoại của LLaVA:

- _Lượt người dùng (chỉ dẫn):_ token ảnh `<image>` kèm câu hỏi tác vụ bằng tiếng Anh, ví dụ với chẩn đoán: "Based on the ECG image, identify which diagnostic statements are present. Briefly explain the key waveform findings, then list each label." Prompt để tiếng Anh nhằm khớp phân phối huấn luyện gốc của PULSE/ECGInstruct (prompt tiếng Việt là off-distribution, làm giảm khả năng hiểu tác vụ).
- _Lượt mô hình (câu trả lời mục tiêu):_ một khối có cấu trúc gồm phần `[REASONING]` (lập luận — để trống ở phiên bản này vì chưa có giám sát lập luận) và phần `[LABELS]` liệt kê từng mã kèm nhãn nhị phân, mỗi dòng dạng `code: 0|1`.

Mô hình được tinh chỉnh (instruction tuning bằng QLoRA, Mục 3.2) để dự đoán câu trả lời mục tiêu này bằng entropy chéo next-token. Như vậy bài toán "phân lớp" được biểu diễn dưới dạng sinh văn bản có cấu trúc theo chỉ dẫn — đồng nhất với cách PULSE đã được huấn luyện — thay vì gắn thêm một đầu phân loại sigmoid ngoài kiến trúc.

*Vì sao dùng khối nhãn số thay vì báo cáo văn xuôi kiểu PULSE.* ECGInstruct của PULSE giám sát mô hình bằng các báo cáo dạng văn xuôi tự do, trong đó chẩn đoán nằm lẫn trong câu chữ @pulse. Khóa luận chủ đích thay bằng một khối nhãn số `code: 0|1` vì ba lý do gắn với mục tiêu đo lường. Thứ nhất, báo cáo văn xuôi chỉ NÊU những bệnh dương và không nhắc tới bệnh âm, nên không biểu diễn được đầy đủ cả 44/12/19 lớp — đặc biệt là phần âm cần thiết cho phân lớp đa nhãn. Thứ hai, một mệnh đề văn xuôi chỉ cho một lần nhắc nhị phân chứ không cho xác suất liên tục theo từng lớp, trong khi macro-AUC — chỉ số chính của khóa luận — đòi hỏi một điểm số liên tục cho mỗi lớp. Thứ ba, một khối nhãn ở vị trí cố định là điều kiện để giao thức teacher-forced (Mục 3.6) đọc xác suất tại đúng ô đã biết; nếu nhãn nằm rải trong văn xuôi thì phải quay lại trích điểm từ sinh tự do, đúng nguồn lỗi đã khiến thử nghiệm ban đầu không trích được điểm ở phần lớn mẫu (Mục 4.4.1). Khối `[REASONING]` được giữ như một chỗ cắm sẵn nhưng để trống ở phiên bản này; hướng kết hợp một khối lập luận văn xuôi (sinh kèm, ví dụ tổng hợp từ siêu dữ liệu PTB-XL) ĐẶT TRƯỚC khối nhãn số — vừa giữ được macro-AUC vừa có diễn giải — được bàn ở Mục 5.2.

*Cân bằng dữ liệu.* Phân phối PTB-XL lệch nặng (NORM khoảng 44%) @ptbxlimb. Để cân bằng và giảm kích thước cho khả thi về tài nguyên, tập huấn luyện áp cơ chế _cap lớp đa số_: giữ TẤT CẢ mẫu thiểu số (có bệnh / đa nhãn), nhưng giới hạn số mẫu thuộc lớp đa số của mỗi tác vụ (NORM-only ở chẩn đoán, SR-only ở nhịp, toàn-âm ở hình thái) ở mức 1500 mẫu mỗi tác vụ; có tùy chọn nhân bản các bản ghi bệnh lý để tăng cường lớp hiếm. Kết quả thu được khoảng 16 nghìn mẫu huấn luyện cân bằng hơn.

Cơ chế này không làm mất thông tin bệnh lý, vì ba lý do. Thứ nhất, cap là kỹ thuật _giảm mẫu lớp đa số_ (random under-sampling) kinh điển cho dữ liệu mất cân bằng, và chỉ áp lên các mẫu THUẦN lớp đa số (ví dụ chỉ-NORM, không kèm bệnh nào khác); toàn bộ mẫu mang nhãn thiểu số được giữ nguyên, nên không có thông tin bệnh lý nào bị loại bỏ. Thứ hai, đây là lựa chọn _chủ đích_ chứ không phải sơ suất: thử nghiệm ban đầu không cân bằng đã sụp đổ về dự đoán toàn NORM (Mục 4.4.1), nên việc hạ tỷ trọng lớp đa số trực tiếp giảm nguy cơ tái diễn hiện tượng này. Thứ ba — và quan trọng nhất — cap CHỈ áp cho tập huấn luyện; tập kiểm định (fold 9) và kiểm thử (fold 10) giữ nguyên phân phối tự nhiên, không cân bằng, nên mọi chỉ số báo cáo đều đo trên phân phối thật và không bị thổi phồng bởi việc cân bằng dữ liệu huấn luyện. Ngưỡng 1500 là siêu tham số; khảo sát có hệ thống ảnh hưởng của ngưỡng này nằm ngoài phạm vi tài nguyên của khóa luận và được để ngỏ như hướng phát triển.

== Phân rã tác vụ và quy tắc gán nhãn

Bài toán được phân rã thành ba tác vụ riêng — chẩn đoán (44), nhịp (12), hình thái (19) — mỗi tác vụ một prompt riêng, khớp với cách Strodthoff vận hành PTB-XL như các tác vụ độc lập @strodthoff và với định dạng ECGInstruct của PULSE @pulse. Quy tắc gán nhãn được tách theo bản chất từng nhóm: với chẩn đoán, dùng ngưỡng likelihood ≥ 50 vì giá trị này chỉ có ý nghĩa cho nhóm chẩn đoán; với nhịp và hình thái, do likelihood luôn bằng 0 @wagner, gán nhãn theo tiêu chí hiện diện. Quy tắc này trực tiếp sửa lỗi của cách tiếp cận áp ngưỡng likelihood đồng loạt, vốn sẽ loại sạch hai nhóm nhịp và hình thái. Prompt tác vụ viết bằng tiếng Anh do PULSE được huấn luyện trên tiếng Anh @pulse.

== Hàm mục tiêu huấn luyện

Hàm mục tiêu trở về entropy chéo next-token nguyên gốc của PULSE, đúng với cơ chế sinh token tự hồi quy của decoder. Tùy chọn thêm là tái trọng số cân bằng lớp theo số mẫu hiệu dụng @cui, với trọng số cho lớp k là $w_k = (1 - beta) / (1 - beta^(n_k))$, trong đó $n_k$ là số mẫu dương của lớp k và $beta$ là siêu tham số cận 1; trọng số này tăng cường đóng góp của lớp hiếm và giảm thống trị của lớp đa số. Focal loss @lin chỉ được khảo sát như một biến thể vì là hàm không proper, có thể làm sai lệch hiệu chuẩn xác suất @mukhoti.

Lý do không dùng một hàm mất mát kiểu phân loại đa nhãn rời rạc là sự không tương thích về cơ chế: loss kiểu đó giả định mỗi nhãn có một logit sigmoid độc lập, trong khi decoder của PULSE sinh token tuần tự với xác suất ràng buộc qua softmax trên toàn từ vựng. Áp một loss như vậy lên xác suất token "1"/"0" của một khối nhãn là cưỡng ép một hàm mất mát sai miền; cộng với một khối nhãn áp đảo bởi token "0" và phân phối lớp mất cân bằng nặng, mô hình dễ rơi vào sụp đổ chế độ và thoái hóa chuỗi sinh @welleck @ptbxlimb.

== Giao thức đánh giá teacher-forced

Đánh giá sinh tự do là một nguồn lỗi lớn: mô hình có thể sinh khối nhãn không hợp lệ nên không trích được điểm. Giao thức teacher-forced thay thế bằng cách tự dựng sẵn khối nhãn chuẩn và đưa cả ảnh lẫn khối này qua một lượt suy luận duy nhất; tại mỗi vị trí token nhãn đã biết của lớp k, đọc xác suất dương bằng softmax hai chiều trên cặp logit của token "1" và token "0":

$ P(y_k = 1) = exp(z["id1"]) / (exp(z["id0"]) + exp(z["id1"])) $

với $z["id1"]$, $z["id0"]$ lần lượt là logit của token "1" và token "0" tại vị trí đó. Vì khối nhãn cố định và mọi vị trí cần chấm đều xác định trước, không còn khâu sinh tự do tạo văn bản méo dạng nên mọi lớp đều có điểm dự đoán.

Về thuật ngữ: tên gọi "teacher-forced" ở đây chỉ một cách đọc xác suất ở thì SUY LUẬN trên một từ vựng giới hạn (chỉ hai token "1"/"0"), thuộc họ _forced-choice / restricted-vocabulary scoring_ — phân biệt với teacher forcing nguyên nghĩa vốn là kỹ thuật ở thì HUẤN LUYỆN (nạp token nhãn thật làm đầu vào bước kế). Sở dĩ mượn tên này vì chuỗi đầu vào tính tới ngay trước ô chấm điểm bị ép cố định. Ngoài ra, $exp(z["id1"]) \/ (exp(z["id0"]) + exp(z["id1"]))$ là xác suất CÓ ĐIỀU KIỆN $P("token"="1" | "token" in {"0","1"})$, tức chuẩn hóa lại chỉ trên hai lựa chọn; giả định ngầm là tại ô nhãn mô hình gần như chỉ phân vân giữa "1" và "0" nên phần khối lượng xác suất bỏ qua là nhỏ.

Về bản chất, teacher-forced là một _giao thức đánh giá_: nó không can thiệp vào trọng số mô hình và không làm mô hình "sinh tốt hơn". Giá trị phương pháp của nó nằm ở chỗ tách hoàn toàn chất lượng phân biệt của mô hình khỏi sự mong manh của khâu sinh chuỗi tự do: nó cho phép đọc một điểm xác suất nhất quán, tái lập và đầy đủ trên _mọi_ lớp — kể cả các lớp hiếm mà đánh giá sinh tự do thường không trích được điểm — qua đó đo được macro-AUC không phụ thuộc ngưỡng trên toàn bộ không gian nhãn. Việc đánh giá sinh tự do ban đầu không trích được điểm ở phần lớn mẫu (Mục 4.4.1) vì vậy chỉ là _bối cảnh_ giải thích vì sao đánh giá sinh tự do không đáng tin cho bài toán này.

Cần nhấn mạnh giao thức này KHÔNG rò rỉ nhãn (label leakage). Ô nhãn của mọi lớp luôn được điền cùng một token chữ số cố định "0" (placeholder), độc lập hoàn toàn với nhãn thật của mẫu; nhãn thật chưa bao giờ được đưa vào chuỗi đầu vào. Xác suất của lớp k được đọc tại logit Ở VỊ TRÍ NGAY TRƯỚC ô số — tức phân phối mà mô hình DỰ ĐOÁN cho ô đó dựa trên ảnh và câu hỏi, chứ không phải token đã bị điền sẵn. Vì khối nhãn là như nhau cho mọi mẫu, nó không mang một bit thông tin nào về nhãn thật của mẫu cụ thể; toàn bộ tín hiệu phân biệt giữa các mẫu chỉ đến từ ảnh ECG đầu vào. Bằng chứng thực nghiệm cho điểm này nằm ở nghiên cứu cắt bỏ ảnh (Mục 4.4.3): do khối nhãn giữ nguyên giữa các điều kiện, nếu nhãn bị rò qua khối answer thì việc xoá hoặc đảo ảnh vẫn phải cho AUC cao; thực tế AUC sụp về mức ngẫu nhiên (~0.5), khẳng định tín hiệu phân biệt đến từ ảnh chứ không từ khối nhãn. Một biến thể kiểm soát đổi token điền từ "0" thành "1" vẫn cho AUC cao hơn hẳn mức ngẫu nhiên (trung bình 0.80 so với 0.50), xác nhận điểm số là phân phối dự đoán của mô hình chứ không phải sao chép token điền sẵn.

Trên cơ sở xác suất này, macro-AUC được tính không phụ thuộc ngưỡng — lấy trung bình trên các lớp có ít nhất một mẫu dương trong tập đánh giá (AUC chỉ xác định khi có cả mẫu dương lẫn âm), nên một số lớp cực hiếm không có mẫu dương trên fold-10 bị loại khỏi trung bình (xem cột "Số lớp có AUC" ở Bảng 4.1). macro-F1 được báo cáo sau khi hiệu chuẩn ngưỡng theo từng lớp trên tập kiểm định fold-9 rồi đóng băng khi chấm fold-10 (tránh rò rỉ thông tin từ tập kiểm thử).

#figure(
  block(width: 100%, inset: (y: 10pt))[
    #set align(center)
    #set par(leading: 0.9em)
    #let cell(body, fill: white) = box(stroke: 0.6pt, inset: (x: 10pt, y: 8pt), radius: 3pt, fill: fill)[#set text(size: 9.5pt); #body]
    #stack(dir: ttb, spacing: 16pt,
      cell([⟨ảnh ECG⟩ + ⟨câu hỏi tác vụ $q_t$⟩ + `[REASONING] … [LABELS]`], fill: luma(238)),
      text(13pt)[↓#h(0.4em)_nối tiếp khối nhãn cố định (token điền "0", độc lập nhãn thật)_],
      grid(columns: 8, column-gutter: 7pt, row-gutter: 7pt, align: horizon,
        cell([`NORM`]), cell([`:`], fill: rgb("#ffe0e6")), cell([`0`], fill: luma(232)),
        cell([`MI`]), cell([`:`], fill: rgb("#ffe0e6")), cell([`0`], fill: luma(232)),
        cell([`…`]), cell([`(K mã)`], fill: luma(245)),
      ),
      text(13pt)[↑#h(0.4em)ô tô hồng = vị trí đọc xác suất],
    )
  ],
  caption: [Hình 3.4: Minh họa khối nhãn teacher-forced và vị trí đọc xác suất. Tại mỗi ô ":" (ngay trước ô số), đọc phân phối token kế tiếp của mô hình và lấy $P(y_k = 1) = "softmax"(z_("id0"), z_("id1"))_1$. Token điền "0" cố định cho mọi mẫu và mọi lớp nên không mang thông tin nhãn thật; toàn bộ tín hiệu phân biệt đến từ ảnh ECG.],
  kind: image,
)
