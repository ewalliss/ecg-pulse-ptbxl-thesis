// ============================================================
//  paper.typ — Báo cáo IEEE-style (tiếng Việt), v2-centered, có so sánh v1 (ASL-Gen).
//  Tác giả: Nguyễn Huỳnh Hải Đăng. Sinh tự động + biên tập. Trích dẫn khớp refs.bib.
// ============================================================
#set document(
  title: "Phan loai benh tim da phuong thuc tren anh ECG: tu VQA cua PULSE sang bo phan lop task-decomposed",
  author: "Nguyen Huynh Hai Dang",
)
#set page(paper: "a4", margin: (x: 1.8cm, y: 2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, lang: "vi")
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "I.1.")
#show heading.where(level: 1): it => block(above: 1.2em, below: 0.6em)[
  #set text(size: 11pt, weight: "bold"); #smallcaps(it)
]
#show heading.where(level: 2): it => block(above: 0.8em, below: 0.4em)[
  #set text(size: 10pt, weight: "bold", style: "italic"); #it.body
]

#align(center)[
  #text(16pt, weight: "bold")[Phân lớp bệnh tim đa phương thức trên ảnh ECG: \
  từ hỏi-đáp thị giác (VQA) của PULSE sang bộ phân lớp task-decomposed trên PTB-XL]
  #v(0.6em)
  #text(11pt)[Nguyễn Huỳnh Hải Đăng]
  #v(0.2em)
  #text(9pt, style: "italic")[Khóa luận tốt nghiệp — Trí tuệ Nhân tạo / Thị giác Máy tính Đa phương thức]
]
#v(0.5em)
#block(inset: (x: 1.2cm))[
  #text(weight: "bold")[Tóm tắt—] Luận văn nghiên cứu bài toán phân loại bệnh tim đa phương thức trên ảnh điện tâm đồ (ECG) thuộc bộ dữ liệu PTB-XL, sử dụng nền tảng PULSE-7B (kiến trúc họ LLaVA-v1.6-Vicuna-7B, gồm bộ mã hoá thị giác CLIP ViT-L/14\@336, lớp chiếu và mô hình ngôn ngữ Vicuna-7B). Khác với baseline một phương thức của Strodthoff chỉ vận hành trên tín hiệu 1-D, mô hình đề xuất nhận đầu vào kết hợp ảnh ECG đi qua bộ mã hoá thị giác và câu hỏi tác vụ đi qua mô hình ngôn ngữ, rồi sinh ra nhãn dạng văn bản @strodthoff @pulse . Phiên bản thử nghiệm ban đầu (v1) áp một hàm mất mát bất đối xứng sinh lên một khối 44 nhãn chẩn đoán duy nhất và đã thất bại: sụp đổ về dự đoán toàn NORM với macro-AUC xấp xỉ 0.5, đồng thời khoảng 73% mẫu cho khối nhãn hỏng khi sinh tự do @ridnik @welleck . Phiên bản v2 khắc phục bằng cách phân rã thành ba tác vụ riêng (chẩn đoán, nhịp, hình thái), giữ nguyên hàm mất mát entropy chéo gốc của PULSE, chỉ tinh chỉnh QLoRA 4-bit và đánh giá theo cơ chế teacher-forced. Trên tập kiểm thử độc lập fold-10 của PTB-XL (2050 ECG mỗi tác vụ), mô hình (tinh chỉnh 2 epoch) đạt macro-AUC trung bình 0.876 (chẩn đoán 0.893, nhịp 0.887, hình thái 0.849) với 0% mẫu hỏng, và vượt chính mô hình nền PULSE-7B chưa fine-tune thêm +0.097 macro-AUC trong so sánh có kiểm soát cùng giao thức; khoảng cách xác suất rõ rệt giữa lớp dương và lớp âm cho thấy mô hình phân biệt thực sự thay vì sụp đổ.
]
#v(0.5em)
#line(length: 100%, stroke: 0.4pt)

= Giới thiệu

Điện tâm đồ là công cụ chẩn đoán tim mạch phổ biến, và PTB-XL là một trong những bộ dữ liệu ECG công khai lớn được chú giải kỹ nhất. Mỗi bản ghi ECG trong PTB-XL được gán nhiều phát biểu SCP, với tổng cộng 71 mã chia thành 44 mã chẩn đoán, 19 mã hình thái và 12 mã nhịp (trong đó 4 mã chồng lấn giữa nhóm chẩn đoán và hình thái) @wagner . Benchmark chuẩn của Strodthoff vận hành PTB-XL như sáu tác vụ riêng biệt, với split cố định 8 fold huấn luyện, fold9 kiểm định và fold10 kiểm thử @strodthoff . Một đặc thù quan trọng của bộ dữ liệu là giá trị likelihood (0-100) chỉ có ý nghĩa cho nhóm chẩn đoán; các mã hình thái và nhịp bị gán likelihood bằng 0, nên nếu áp một ngưỡng likelihood lớn hơn hoặc bằng 50 đồng loạt thì toàn bộ nhóm hình thái và nhịp sẽ bị loại bỏ @wagner .

PULSE-7B bản chất là một mô hình hỏi đáp thị giác (VQA) sinh văn bản, không phải một bộ phân lớp đa nhãn được hiệu chuẩn xác suất. Trong chế độ đánh giá sinh tự do, mô hình có thể trả về văn bản không trích được điểm số, hoặc rơi vào trạng thái lặp và sinh khối nhãn hỏng. Hơn nữa, hàm mất mát entropy chéo per-token trên một khối phần lớn là ký tự "0" cùng với mất cân bằng lớp nặng dễ đẩy mô hình tới hiện tượng sụp đổ chế độ, tức luôn dự đoán lớp đa số NORM (chiếm khoảng 44% PTB-XL) @welleck @ptbxlimb . Đây chính là điều quan sát được ở v1 khi cố ép một hàm mất mát bất đối xứng vốn được định nghĩa cho head sigmoid K-logit độc lập vào một bộ giải mã sinh token, gây lệch kiến trúc @ridnik .

Việc phân rã theo tác vụ là cần thiết vì nhiều lý do. Thứ nhất, nó tách bạch quy tắc gán nhãn khác nhau giữa các nhóm: nhóm chẩn đoán dùng ngưỡng likelihood lớn hơn hoặc bằng 50, còn nhóm nhịp và hình thái dùng quy tắc hiện diện (presence) do likelihood bằng 0, qua đó sửa lỗi v1 nuốt mất nhóm hình thái và nhịp @wagner . Thứ hai, nó khớp với cấu trúc sáu tác vụ của Strodthoff và định dạng ECGInstruct của PULSE, mỗi tác vụ ứng với một prompt riêng @strodthoff @pulse . Thứ ba, nó cho phép đánh giá teacher-forced ổn định: tự dựng khối nhãn rồi đọc xác suất P(y bằng 1) tại các vị trí đã biết trong một lượt forward, loại bỏ hoàn toàn hiện tượng mẫu hỏng.

Các đóng góp chính của luận văn gồm:
- Chỉ ra và phân tích nguyên nhân thất bại của cách tiếp cận một khối 44 nhãn với hàm mất mát bất đối xứng sinh (v1), bao gồm lệch kiến trúc giữa loss sigmoid độc lập và bộ giải mã sinh token cùng hiện tượng sụp đổ chế độ @ridnik @welleck .
- Đề xuất khung phân rã ba tác vụ (chẩn đoán 44 nhãn, nhịp 12 nhãn, hình thái 19 nhãn) với quy tắc gán nhãn riêng cho từng nhóm, giữ nguyên lõi PULSE và hàm mất mát entropy chéo gốc, chỉ tinh chỉnh QLoRA 4-bit (r bằng 16, alpha bằng 32) trên LLM trong khi đông cứng bộ mã hoá thị giác @pulse @cui .
- Thiết kế quy trình đánh giá teacher-forced đọc xác suất per-class trong một lượt forward, đạt 0% mẫu hỏng, cho phép tính macro-AUC không phụ thuộc ngưỡng và hiệu chuẩn ngưỡng per-class.
- Đạt macro-AUC trung bình 0.876 trên tập kiểm thử độc lập fold-10 PTB-XL (chẩn đoán 0.893, nhịp 0.887, hình thái 0.849), vượt chính mô hình nền PULSE-7B +0.097 trong so sánh có kiểm soát, thu hẹp khoảng cách với baseline tín hiệu 1-D của Strodthoff trong khi giữ tính đa phương thức @strodthoff @gem .

= Công trình liên quan

Mục này đặt nghiên cứu vào hai dòng công trình: (a) benchmark PTB-XL cùng quy ước nhãn và sơ đồ sáu tác vụ của Strodthoff; (b) các mô hình ngôn ngữ-thị giác cho điện tâm đồ (ECG-VLM) cùng các hệ thống ECG đa phương thức liên quan. Mục tiêu là làm rõ vị trí của luận văn: một mô hình đa phương thức trên ẢNH ECG, nền PULSE-7B, và lý do vì sao công thức huấn luyện thắng cuộc là cross-entropy next-token trên prompt phân rã theo tác vụ chứ không phải một hàm mất mát bất đối xứng kiểu phân loại đa nhãn.

Benchmark PTB-XL và quy ước nhãn. PTB-XL là bộ dữ liệu ECG 12 đạo trình quy mô lớn, trong đó mỗi bản ghi được gán nhiều phát biểu SCP đồng thời, tạo nên bài toán phân loại đa nhãn thực thụ @wagner . Bộ 71 mã SCP được tổ chức thành ba nhóm: 44 mã chẩn đoán (diagnostic), 19 mã hình thái (form) và 12 mã nhịp (rhythm), trong đó có 4 mã chồng lấn giữa diagnostic và form @wagner . Một đặc thù quan trọng của quy ước nhãn là giá trị likelihood (thang 0-100) chỉ mang ý nghĩa cho nhóm chẩn đoán; các mã form và rhythm bị gán likelihood bằng 0. Hệ quả là nếu áp một ngưỡng likelihood lớn hơn hoặc bằng 50 đồng loạt cho toàn bộ nhãn thì sẽ loại sạch nhóm form và rhythm, một cái bẫy quy ước mà mọi pipeline gán nhãn phải xử lý tách biệt. Để xây dựng một benchmark có thể so sánh được, Strodthoff và cộng sự vận hành PTB-XL như SÁU tác vụ riêng biệt (trong đó có ba tác vụ trung tâm: chẩn đoán, nhịp, hình thái), với split chuẩn gồm 8 fold huấn luyện, fold 9 để kiểm định và fold 10 để kiểm thử @strodthoff . Đường biên trên (ceiling) do mạng CNN 1-D của Strodthoff thiết lập trên tín hiệu thô đạt macro-AUC 0.937 cho chẩn đoán, 0.957 cho nhịp và 0.896 cho hình thái @strodthoff . Đây là baseline đơn phương thức: chỉ dùng tín hiệu 1-D, không dùng ảnh và không dùng text. Luận văn này khác biệt ở chỗ đầu vào là ẢNH ECG đi qua bộ mã hóa thị giác cộng với câu hỏi tác vụ dạng text đi qua LLM, còn đầu ra là text (nhãn, có thể kèm lập luận).

Các mô hình ngôn ngữ-thị giác và đa phương thức cho ECG. PULSE là mô hình nền của luận văn, dạy LLaVA-v1.6-Vicuna-7B đọc hiểu ẢNH điện tâm đồ qua tập chỉ dẫn ECGInstruct; PULSE huấn luyện bằng cross-entropy next-token tiêu chuẩn của decoder và là mô hình phân rã bài toán thành các tác vụ chỉ dẫn riêng @pulse . Trên PTB-XL Super (5 lớp), PULSE-7B zero-shot đạt macro-AUC 82.4, vượt xa GPT-4o (55.6) và chính backbone LLaVA-1.6 chưa tinh chỉnh (50.0, tức gần như ngẫu nhiên) @pulse ; điều này cho thấy việc tinh chỉnh theo chỉ dẫn ECG là then chốt để backbone thị giác tổng quát trở nên dùng được cho ECG. GEM là hệ đa phương thức kết hợp cả ẢNH, TÍN HIỆU và TEXT, đạt 83.4 trên PTB-XL Super, nhỉnh hơn PULSE nhờ bổ sung kênh tín hiệu @gem . ECG-Chat đi theo hướng hội thoại và sinh báo cáo, ghép biểu diễn ECG với LLM để trả lời câu hỏi và lập luận lâm sàng @ecgchat . MERL học biểu diễn ECG bằng tiền huấn luyện tương phản giữa tín hiệu và báo cáo (đa phương thức tín hiệu-text), hỗ trợ phân loại zero-shot qua truy vấn nhãn dạng văn bản @merl . HeartLang xây dựng một mô hình nền cho tín hiệu ECG theo lối "ngôn ngữ của nhịp tim", học biểu diễn tự giám sát ở mức nhịp/đoạn rồi chuyển giao xuống các tác vụ phân loại @heartlang . Tang và cộng sự đóng góp vào dòng ECG-VLM/đa phương thức tập trung vào căn chỉnh biểu diễn và sinh mô tả @tang . Nandakishor khai thác một LLM (họ Llama) cho ECG, tiếp cận bài toán như sinh văn bản chẩn đoán thay vì gắn một đầu phân loại sigmoid @nandakishor . ECG-QA định dạng ECG thành bài toán hỏi-đáp, biến chẩn đoán thành trả lời câu hỏi tự nhiên và do đó cũng dựa trên mục tiêu sinh token @ecgqa .

Điểm chung mang tính quyết định: KHÔNG có mô hình ECG-VLM hay ECG đa phương thức nào trong số trên sử dụng hàm mất mát bất đối xứng (ASL) làm mục tiêu huấn luyện. ASL được định nghĩa cho một đầu phân loại sigmoid gồm K logit ĐỘC LẬP, với tổng mất mát theo từng nhãn, hệ số tập trung phía âm lớn hơn phía dương và một biên xác suất @ridnik ; đây là thiết kế cho head phân loại đa nhãn, không phải cho decoder sinh token tuần tự. Các hệ sinh-ngôn ngữ ở trên đều dùng cross-entropy next-token, tức tối ưu xác suất chuỗi token đầu ra theo kiểu tự hồi quy @pulse @ecgchat @nandakishor @ecgqa . Quan sát này lý giải trực tiếp thất bại của phiên bản v1 trong luận văn: áp ASL-Gen lên xác suất token "1"/"0" của một khối 44 nhãn chẩn đoán duy nhất là sự lệch kiến trúc giữa giả định "K logit độc lập" của ASL và bản chất decoder, dẫn tới sụp đổ về dự đoán toàn NORM (macro-AUC khoảng 0.5) và khối nhãn hỏng khi sinh tự do. Trên nền tảng lý thuyết, cross-entropy per-token áp lên một khối phần lớn là "0" trong điều kiện mất cân bằng dễ gây mode-collapse và suy thoái sinh @welleck , còn việc tỷ lệ NORM của PTB-XL xấp xỉ 44% khiến đường tắt "đoán toàn âm" càng hấp dẫn @ptbxlimb .

Công thức thắng cuộc mà luận văn rút ra, nhất quán với toàn bộ dòng công trình trên, là giữ nguyên cross-entropy next-token GỐC của PULSE và chuyển toàn bộ gánh nặng cân bằng lớp sang phía dữ liệu và bố cục prompt: phân rã thành ba tác vụ riêng (chẩn đoán 44, nhịp 12, hình thái 19) theo đúng tinh thần sáu tác vụ của Strodthoff @strodthoff  và ECGInstruct của PULSE @pulse ; tách quy tắc nhãn theo từng nhóm (chẩn đoán dùng ngưỡng likelihood, còn nhịp và hình thái dùng sự hiện diện vì likelihood bằng 0) @wagner ; và chỉ bổ sung tùy chọn tái trọng số cân bằng theo lớp dạng class-balanced @cui , với focal chỉ đóng vai trò ablation @lin  và lưu ý focal không phải hàm mất mát "proper" cho hiệu chuẩn xác suất @mukhoti . Nói cách khác, đóng góp không nằm ở một hàm mất mát mới mà ở việc tái cấu trúc bài toán cho khớp với cả quy ước PTB-XL lẫn kiến trúc sinh của PULSE.

Bảng dưới tóm tắt so sánh các mô hình theo phương thức đầu vào, mục tiêu huấn luyện, cách định nghĩa nhãn và khả năng lập luận.

#figure(
  table(
    columns: 5,
    inset: 4pt,
    align: (left,) + (center,) * 4,
    table.header([*Mô hình*], [*Phương thức*], [*Mục tiêu huấn luyện*], [*Định nghĩa nhãn*], [*Lập luận text*]),
    [Strodthoff (CNN 1-D) @strodthoff ], [tín hiệu], [BCE đa nhãn, head sigmoid], [đa nhãn theo 6 tác vụ], [không],
    [pulse @pulse ], [ảnh + text], [CE next-token], [chỉ dẫn ECGInstruct], [có],
    [gem @gem ], [ảnh + tín hiệu + text], [CE next-token (sinh)], [chỉ dẫn đa phương thức], [có],
    [ecgchat @ecgchat ], [tín hiệu + text], [CE next-token (hội thoại)], [hỏi-đáp / báo cáo], [có],
    [merl @merl ], [tín hiệu + text], [tương phản tín hiệu-text], [truy vấn nhãn zero-shot], [hạn chế],
    [heartlang @heartlang ], [tín hiệu], [tự giám sát + tinh chỉnh], [nhãn tác vụ hạ nguồn], [không],
    [nandakishor @nandakishor ], [tín hiệu/text (LLM)], [CE next-token], [sinh văn bản chẩn đoán], [có],
    [ecgqa @ecgqa ], [tín hiệu + text], [CE next-token], [hỏi-đáp], [có],
    [Luận văn (PULSE-7B + QLoRA)], [ảnh + text], [CE next-token, prompt phân rã tác vụ], [likelihood (diag) / hiện diện (rhythm, form)], [có],
  ),
  caption: [So sánh tóm tắt một số mô hình ECG đa phương thức và benchmark liên quan.],
)


Tổng hợp lại, không có công trình nào trong bảng dùng ASL; tất cả các hệ sinh-ngôn ngữ đều dựa trên cross-entropy next-token, và luận văn kế thừa đúng mục tiêu đó trong khi bổ sung bố cục prompt phân rã ba tác vụ cùng quy tắc nhãn riêng cho từng nhóm để tôn trọng đặc thù likelihood của PTB-XL @wagner @strodthoff @pulse .

= Phiên bản thử nghiệm v1 (ASL-Gen) và phân tích thất bại

Phiên bản thử nghiệm đầu tiên (v1) được thiết kế theo hướng coi toàn bộ bài toán chẩn đoán là một bài phân loại đa nhãn duy nhất, đặt trực tiếp lên trên kiến trúc sinh của PULSE-7B. Cụ thể, mô hình được yêu cầu sinh ra MỘT khối nhãn gồm 44 mã chẩn đoán (diagnostic) liền nhau, mỗi mã được biểu diễn bằng một token nhị phân "1" hoặc "0" cho biết bệnh tương ứng có mặt hay không. Tín hiệu giám sát không phải là entropy chéo next-token gốc của PULSE mà là một hàm mất mát bất đối xứng phiên bản sinh (gọi là ASL-Gen), áp lên xác suất của token "1"/"0" tại từng vị trí trong khối 44 nhãn. Ý tưởng đằng sau là vay mượn ưu thế của Asymmetric Loss trong phân loại đa nhãn mất cân bằng, kỳ vọng nó sẽ trừng phạt mạnh các âm tính giả của lớp hiếm và giảm áp lực từ khối lượng lớn nhãn "0".

Vấn đề cốt lõi nằm ở sự lệch kiến trúc giữa định nghĩa gốc của ASL và bản chất của một decoder sinh token. Asymmetric Loss được Ridnik và cộng sự xây dựng cho một head phân loại sigmoid với K logit ĐỘC LẬP @ridnik : với mỗi nhãn k, mô hình xuất một logit z\_k, đi qua hàm sigmoid để cho xác suất p\_k = sigma(z\_k), và mất mát tổng là L = tổng theo k của L(sigma(z\_k), y\_k). Trong công thức này, nhánh dương dùng (1 - p)^gamma\_pos nhân log(p) còn nhánh âm dùng p\_m^gamma\_neg nhân log(1 - p\_m), với p\_m là xác suất đã dịch một biên độ m (probability shifting, p\_m = max(p - m, 0)) và gamma\_neg lớn hơn gamma\_pos để giảm trọng số mẫu âm dễ. Ba giả định nền tảng của thiết kế này — K đầu ra sigmoid độc lập, mỗi nhãn có một logit riêng được hiệu chỉnh xác suất tách biệt, và sự bất đối xứng cùng biên độ m tác động trực tiếp lên không gian xác suất của head — đều KHÔNG tồn tại trong decoder của PULSE. Ở đó, mỗi vị trí trong khối nhãn là một phân phối softmax trên TOÀN BỘ từ vựng, các token được sinh tuần tự và phụ thuộc lẫn nhau theo chuỗi, không phải K quyết định nhị phân song song. Ép ASL lên xác suất token "1"/"0" do đó là áp một hàm mất mát của head phân loại độc lập lên một cơ chế sinh hoàn toàn khác về mặt thống kê và quy nạp; biên độ m và sự bất đối xứng gamma\_neg/gamma\_pos mất đi ngữ nghĩa vốn có khi không còn K logit sigmoid riêng biệt để tác động.

Sự lệch kiến trúc này kết hợp với hai yếu tố làm trầm trọng thêm tình trạng sụp đổ. Thứ nhất, khối 44 nhãn áp đảo bởi token "0" (mỗi ECG chỉ dương tính ở một số ít mã), cộng với phân phối lớp cực kỳ mất cân bằng của PTB-XL — trong đó NORM chiếm khoảng 44% số mẫu @ptbxlimb  — tạo ra sức ép cực lớn đẩy mô hình về lời giải tầm thường. Thứ hai, ngay cả entropy chéo per-token trên một chuỗi mục tiêu chủ yếu là "0" cũng dễ dẫn tới mode-collapse: mô hình sinh học cách lặp lại đầu ra tần suất cao và bỏ qua tín hiệu hiếm, đúng với cơ chế suy thoái và lặp lại của giải mã sinh mà Welleck và cộng sự đã phân tích @welleck . Như vậy ASL-Gen vừa không cung cấp đúng dạng giám sát mà ASL được thiết kế để phát huy, vừa không khắc phục được xu hướng suy thoái nội tại của decoder.

Hậu quả thể hiện qua hai triệu chứng quan sát được, hỏng trên hai trục khác nhau. Trục thứ nhất là SỤP ĐỔ phân loại: mô hình hội tụ về việc dự đoán gần như toàn NORM, khiến macro-AUC trên tập chẩn đoán rơi về khoảng 0.5, tức không tốt hơn đoán ngẫu nhiên và không phân biệt được bệnh nào. Trục thứ hai là HỎNG ĐỊNH DẠNG khi đánh giá bằng sinh tự do: khoảng 73% số mẫu cho ra khối nhãn không hợp lệ — hoặc không chứa chữ số có thể bóc tách (digits = 0), hoặc rơi vào vòng lặp sinh tới 275 dòng — nên không thể trích được điểm nhãn từ đầu ra. Hai triệu chứng này cho thấy thiết kế v1 thất bại đồng thời ở khả năng học phân biệt và ở khả năng sinh ra cấu trúc đầu ra ổn định, và chính cặp thất bại đó trở thành mỏ neo so sánh cho các sửa đổi của phiên bản v2: phân rã theo task, quay về entropy chéo gốc của PULSE, và đánh giá teacher-forced để loại bỏ hoàn toàn lỗi định dạng.

= Phương pháp đề xuất (v2)

Phương pháp v2 giữ nguyên lõi đa phương thức của PULSE-7B và chỉ can thiệp ở các tầng định dạng bài toán, gán nhãn, hàm mục tiêu và giao thức đánh giá, qua đó khắc phục triệt để hai trục thất bại của v1.

Kiến trúc đa phương thức và chiến lược tinh chỉnh

Mô hình kế thừa kiến trúc họ LLaVA-v1.6-Vicuna-7B của PULSE-7B @pulse : ảnh ECG được đưa qua bộ mã hóa thị giác CLIP ViT-L/14 ở độ phân giải 336, các đặc trưng thị giác được chiếu sang không gian biểu diễn của mô hình ngôn ngữ thông qua một bộ chiếu (projector), rồi nối với chuỗi token văn bản của câu hỏi task để mô hình ngôn ngữ Vicuna-7B sinh đáp án dưới dạng văn bản. Đây là điểm phân biệt cốt lõi với baseline Strodthoff vốn đơn phương thức, chỉ xử lý tín hiệu một chiều @strodthoff : ở đây đầu vào gồm hai luồng - thị giác (ảnh ECG qua CLIP) và ngôn ngữ (câu hỏi task qua LLM) - còn đầu ra là văn bản nhãn (có thể kèm lập luận).

Về tinh chỉnh, chúng tôi đông cứng toàn bộ nhánh thị giác (CLIP và projector) và chỉ huấn luyện mô hình ngôn ngữ bằng QLoRA 4-bit với r=16, alpha=32; không sửa đổi lõi PULSE. Lựa chọn này vừa tiết kiệm tài nguyên (toàn bộ thí nghiệm chạy được trên một GPU V100), vừa bảo toàn năng lực biểu diễn ảnh ECG đã được PULSE học sẵn, tránh phá vỡ sự căn chỉnh thị giác - ngôn ngữ vốn có.

Phân rã ba task và quy tắc gán nhãn

PTB-XL gán cho mỗi ECG nhiều phát biểu SCP, với 71 mã chia thành 44 mã chẩn đoán (diagnostic), 19 mã hình thái (form) và 12 mã nhịp (rhythm), trong đó có 4 mã chồng lấn giữa diagnostic và form @wagner . Sai lầm kiến trúc của v1 là dồn toàn bộ vào một khối 44 nhãn chẩn đoán duy nhất. Trong v2, chúng tôi phân rã thành ba task riêng biệt - diag (44), rhythm (12), form (19) - mỗi task dùng một prompt riêng, ăn khớp với cách benchmark Strodthoff vận hành PTB-XL như các task độc lập @strodthoff  và với định dạng ECGInstruct của PULSE @pulse .

Quy tắc gán nhãn được tách theo bản chất từng nhóm mã. Với task diag, ta dùng ngưỡng likelihood>=50 vì giá trị likelihood (0-100) chỉ có ý nghĩa cho nhóm chẩn đoán. Với rhythm và form, do likelihood luôn bằng 0 @wagner , ta gán nhãn theo tiêu chí hiện diện (presence): chỉ cần mã xuất hiện trong chú giải là dương. Quy tắc này trực tiếp sửa lỗi nghiêm trọng của v1: nếu áp đồng loạt ngưỡng likelihood>=50, toàn bộ nhãn form và rhythm sẽ bị loại sạch (vì likelihood=0), tức v1 đã vô tình nuốt mất hai nhóm mã này.

Hàm mất mát: cross-entropy có tái trọng số cân bằng lớp, và lý do loại bỏ ASL

Hàm mục tiêu của v2 trở về cross-entropy next-token gốc của PULSE, đúng với cơ chế sinh token tự hồi quy của decoder. Tùy chọn thêm là tái trọng số cân bằng lớp (class-balanced reweighting) @cui  với trọng số cho lớp k:

w\_k = (1 - beta) / (1 - beta^{n\_k})

trong đó n\_k là số mẫu dương của lớp k và beta là siêu tham số trong khoảng cận 1; trọng số này tăng cường đóng góp của các lớp hiếm và giảm thống trị của lớp đa số. Mất mát focal @lin  chỉ được khảo sát như một biến thể (ablation), vì nó là hàm không proper, có thể làm sai lệch hiệu chuẩn xác suất @mukhoti .

Lý do loại bỏ ASL-Gen là lệch kiến trúc. ASL nguyên thủy được định nghĩa cho một đầu phân loại sigmoid gồm K logit độc lập, dạng L = tổng theo k của L(sigma(z\_k), y\_k) với gamma\_neg > gamma\_pos và biên m @ridnik ; nó giả định mỗi nhãn có một logit riêng và một hàm sigmoid riêng. Decoder sinh token của PULSE không có cấu trúc đó: nó sinh tuần tự các token và xác suất bị ràng buộc qua softmax trên toàn từ vựng, nên việc áp ASL lên xác suất token "1"/"0" của một khối nhãn là cưỡng ép một hàm mất mát sai miền. Hệ quả quan sát ở v1 đúng như dự đoán lý thuyết: cross-entropy per-token trên một khối token áp đảo bởi ký tự "0" cộng với mất cân bằng lớp đẩy mô hình vào sụp đổ chế độ (mode-collapse), dự đoán toàn NORM với macro-AUC xấp xỉ 0.5 @welleck , trong khi NORM chiếm khoảng 44% PTB-XL @ptbxlimb . Trở về cross-entropy gốc cộng tái trọng số cân bằng lớp loại bỏ cả mâu thuẫn kiến trúc lẫn áp lực sụp đổ này.

Định dạng đầu vào/đầu ra kèm lập luận

Mỗi mẫu huấn luyện gồm ảnh ECG và một prompt task bằng tiếng Anh (vì PULSE được huấn luyện trên tiếng Anh @pulse ); đầu ra là khối nhãn dạng văn bản. Phần lập luận (reasoning) kèm theo đáp án hiện được để trống trong phiên bản này.

Cân bằng dữ liệu được thực hiện bằng cách giới hạn (cap) lớp đa số ở 1500 mẫu mỗi task, đưa tập huấn luyện về khoảng 16k mẫu, đủ nhỏ để huấn luyện khả thi mà vẫn giữ tín hiệu các lớp thiểu số.

Giao thức đánh giá teacher-forced

Đánh giá tự do (free-form generation) là nguyên nhân thứ hai khiến v1 thất bại: khoảng 73% mẫu cho khối nhãn hỏng (digits bằng 0 hoặc lặp tới 275 dòng) nên không trích được điểm. v2 thay bằng giao thức teacher-forced. Ta tự dựng sẵn khối nhãn chuẩn và đưa cả ảnh lẫn khối này qua một forward pass duy nhất; tại mỗi vị trí token nhãn đã biết của lớp k, ta đọc xác suất dương bằng cách lấy softmax hai chiều trên cặp logit của token "1" và token "0":

P(y\_k = 1) = softmax([z[id0], z[id1]])

tức P(y\_k=1) bằng exp(z[id1]) / (exp(z[id0]) + exp(z[id1])), với z[id1], z[id0] lần lượt là logit của token "1" và token "0" tại vị trí đó. Vì khối nhãn được dựng cố định và mọi vị trí cần chấm đều xác định trước, không còn khâu sinh tự do để mô hình tạo ra văn bản méo dạng; do đó tỷ lệ mẫu hỏng là 0% và mọi lớp đều có điểm dự đoán. Giao thức này không rò rỉ nhãn: ô nhãn luôn điền token "0" cố định độc lập với nhãn thật của mẫu, còn xác suất được đọc tại logit DỰ ĐOÁN ngay trước ô đó nên nhãn thật không vào context. Trên cơ sở xác suất này, macro-AUC được tính không cần ngưỡng (threshold-free) — lấy trung bình trên các lớp có ít nhất một mẫu dương trong tập đánh giá — còn macro-F1 được báo cáo sau khi hiệu chuẩn ngưỡng theo từng lớp (per-class).

= Thực nghiệm và kết quả

Thiết lập thực nghiệm

Mô hình được xây dựng trên nền PULSE-7B (LLaVA-v1.6-Vicuna-7B), gồm bộ mã hóa thị giác CLIP ViT-L/14\@336, lớp chiếu (projector) và mô hình ngôn ngữ Vicuna-7B @pulse . Đây là bài toán đa phương thức: đầu vào là ảnh ECG (đưa qua CLIP) cùng văn bản câu hỏi của từng task (đưa qua LLM), đầu ra là văn bản nhãn. Chúng tôi đông cứng toàn bộ nhánh thị giác và projector, chỉ tinh chỉnh LLM bằng QLoRA 4-bit với r=16, alpha=32, learning rate 2e-4, độ chính xác fp16, attention theo cơ chế sdpa, huấn luyện 2 epoch. Lõi PULSE không bị sửa đổi; hàm mất mát giữ nguyên cross-entropy next-token gốc của PULSE @pulse , có tùy chọn class-balanced reweight w\_k=(1-beta)/(1-beta^{n\_k}) @cui , với focal chỉ đóng vai trò ablation @lin .

Bài toán PTB-XL được phân rã thành ba task riêng biệt, mỗi task một prompt: chẩn đoán diagnostic (44 mã), nhịp rhythm (12 mã) và hình thái form (19 mã), khớp với cách Strodthoff vận hành PTB-XL như các task độc lập @strodthoff  và với ECGInstruct của PULSE @pulse . Quy tắc gán nhãn tuân theo cấu trúc dữ liệu: task diagnostic dùng ngưỡng likelihood>=50, còn rhythm và form dùng quy tắc hiện diện (presence) vì các mã này bị gán likelihood=0 @wagner . Dữ liệu huấn luyện được cân bằng bằng cách giới hạn lớp đa số ở mức 1500 mẫu mỗi task, thu được khoảng 16 nghìn mẫu. Prompt task viết bằng tiếng Anh do PULSE được huấn luyện trên tiếng Anh @pulse .

Đánh giá được thực hiện theo cơ chế teacher-forced trên cả tập kiểm định fold-9 (2034 ECG mỗi task) và tập kiểm thử độc lập fold-10 (2050 ECG mỗi task) theo split chuẩn của Strodthoff @strodthoff : khối nhãn được dựng sẵn, xác suất P(y\_k=1)=softmax([z[id0],z[id1]]) được đọc tại vị trí token đã biết trong một forward pass duy nhất. Cách này cho phép chấm điểm mọi lớp, không phụ thuộc khả năng sinh tự do của mô hình, đồng thời cho macro-AUC không phụ thuộc ngưỡng kèm hiệu chuẩn ngưỡng per-class. Fold-10 là số headline không thiên lệch vì hoàn toàn không tham gia huấn luyện hay hiệu chuẩn.

Kết quả per-task

#figure(
  table(
    columns: 6,
    inset: 4pt,
    align: (left,) + (center,) * 5,
    table.header([*Task*], [*Số lớp có AUC*], [*macro-AUC fold-10 (test)*], [*macro-F1 (val-frozen)*], [*mean P(1) dương*], [*mean P(1) âm*]),
    [diagnostic (44)], [44/44], [0.893], [0.24], [0.44], [0.03],
    [rhythm (12)], [11/12], [0.887], [0.33], [0.64], [0.06],
    [form (19)], [19/19], [0.849], [0.22], [0.35], [0.05],
    [Trung bình], [-], [0.876], [0.26], [-], [-],
  ),
  caption: [Kết quả v2 (tinh chỉnh 2 epoch) theo từng tác vụ trên PTB-XL fold-10 (kiểm thử, 2050 ECG/tác vụ), eval teacher-forced. macro-AUC lấy trung bình trên các lớp có mẫu dương (cột "Số lớp có AUC"); macro-F1 dùng ngưỡng per-class hiệu chuẩn trên fold-9 rồi đóng băng khi chấm fold-10.],
)


Trên tập kiểm thử fold-10, ở cả ba task mean P(1) của lớp dương lớn hơn hẳn lớp âm (diagnostic 0.44 so với 0.03; rhythm 0.64 so với 0.06; form 0.35 so với 0.05), cho thấy mô hình thực sự phân biệt được nhãn có và không, không rơi vào sụp đổ chế độ. Tỷ lệ mẫu cho khối nhãn hỏng (malformed) là 0%; macro-AUC được lấy trung bình trên các lớp có ít nhất một mẫu dương trong fold-10 (diagnostic 44/44, rhythm 11/12, form 19/19 — lớp nhịp còn lại không có mẫu dương nên AUC không xác định). Những lớp tốt nhất trên test gồm clbbb 1.00, injin/injil 1.00 và crbbb/3avb 0.99 (diagnostic), stach và afib 0.99 (rhythm), pvc 0.99 và prc(s) 0.98 (form). Việc huấn luyện thêm cải thiện đều trên cả ba task: macro-AUC trung bình fold-10 tăng từ 0.818 (epoch 1) lên 0.876 (epoch 2), chưa có dấu hiệu khớp quá, nên các con số nên hiểu là cận dưới.

So sánh v1, v2 và các mốc tham chiếu

#figure(
  table(
    columns: 4,
    inset: 4pt,
    align: (left,) + (center,) * 3,
    table.header([*Hệ thống*], [*Phương thức*], [*macro-AUC*], [*Tỷ lệ malformed*]),
    [v1 (ASL-Gen, thử nghiệm)], [ảnh+text], [~0.5 (sụp về toàn NORM)], [~73%],
    [v2 (đề xuất, test fold-10)], [ảnh+text], [0.876 (TB); diag 0.893 / rhythm 0.887 / form 0.849], [0%],
    [PULSE-7B base zero-shot (test fold-10, cùng eval)], [ảnh+text], [0.779 (TB); diag 0.835 / rhythm 0.774 / form 0.728], [0%],
    [Strodthoff CNN 1-D (trần)], [tín hiệu], [diag 0.937 / rhythm 0.957 / form 0.896], [-],
    [PULSE-7B zero-shot (Super, 5 lớp, theo giấy gốc)], [ảnh+text], [0.824], [-],
    [GPT-4o zero-shot (Super)], [ảnh+text], [0.556], [-],
    [LLaVA-1.6 backbone zero-shot (Super)], [ảnh+text], [0.500], [-],
    [GEM (ảnh+tín hiệu+text, Super)], [đa phương thức], [0.834], [-],
  ),
  caption: [So sánh v1, v2 và các mốc tham chiếu (macro-AUC; tỷ lệ mẫu hỏng).],
)


Phiên bản v1 áp một loss bất đối xứng sinh (ASL-Gen) lên xác suất token "1"/"0" của một khối 44 nhãn chẩn đoán duy nhất và thất bại trên hai trục: sụp đổ về dự đoán toàn NORM khiến macro-AUC chỉ còn ~0.5, và khi đánh giá bằng sinh tự do thì khoảng 73% mẫu cho khối nhãn hỏng nên không trích được điểm. Nguyên nhân là ASL vốn được định nghĩa cho head sigmoid K-logit độc lập với gamma\_neg lớn hơn gamma\_pos và margin m @ridnik , không phù hợp với decoder sinh token, cộng với cross-entropy per-token trên khối phần lớn là "0" giữa bối cảnh mất cân bằng (NORM chiếm khoảng 44% PTB-XL @ptbxlimb ) dẫn tới sụp đổ chế độ @welleck . Việc phân rã ba task và sửa quy tắc nhãn cho rhythm/form đã khắc phục cả hai lỗi này.

Phân tích

Kết quả trên tập kiểm thử fold-10 khẳng định mô hình phân biệt thật chứ không sụp đổ: khoảng cách xác suất giữa lớp dương và lớp âm rõ rệt ở cả ba task, và 0% mẫu hỏng. So với mốc trần do CNN 1-D của Strodthoff thiết lập (diag 0.937, rhythm 0.957, form 0.896) @strodthoff , v2 vẫn còn khoảng cách nhưng tiệm cận đáng kể dù chỉ dùng ảnh ECG (không có tín hiệu 1-D gốc) và chỉ tinh chỉnh nhẹ QLoRA 2 epoch: diagnostic đạt 0.893, rhythm 0.887, form 0.849. Cần nhấn mạnh một điểm so sánh công bằng: con số headline 0.876 của v2 được đo trên TOÀN BỘ tập nhãn chi tiết (44+12+19 mã), khó hơn nhiều so với mức PTB-XL Super chỉ 5 lớp mà PULSE-7B zero-shot (0.824) và GEM (0.834) báo cáo @pulse @gem ; do đó về trị tuyệt đối 0.876 đã vượt nhẹ hai mốc Super đó dù giải một bài toán khó hơn rõ rệt, và riêng tác vụ diagnostic của v2 (0.893) đã vượt các mốc Super này. So với các backbone tổng quát chưa tinh chỉnh ECG, v2 vượt xa GPT-4o (0.556) và LLaVA-1.6 (0.500) @pulse . Điều này cho thấy việc tinh chỉnh nhẹ nhánh ngôn ngữ trên dữ liệu cân bằng đủ để nâng năng lực phân loại từ ảnh ECG.

Cần lưu ý một caveat về macro-F1: với ngưỡng per-class hiệu chuẩn trên fold-9 rồi đóng băng khi chấm fold-10 (tránh chọn ngưỡng trên chính tập kiểm thử), macro-F1 trên test vẫn thấp (khoảng 0.22–0.33; diag 0.24, rhythm 0.33, form 0.22) do phép trung bình tính đều cả các lớp hiếm đặc thù của PTB-XL; vì vậy chúng tôi lấy macro-AUC làm chỉ số chính (headline) bởi nó không phụ thuộc ngưỡng và phản ánh khả năng phân biệt thực. macro-AUC cao thể hiện khả năng XẾP HẠNG/phân biệt, chưa đồng nghĩa một bộ phân lớp đã hiệu chuẩn sẵn sàng triển khai. Trong ba task, form yếu nhất (0.849), với lớp tab chỉ đạt khoảng 0.34, phù hợp với mức trần thấp hơn của task này @strodthoff  và bản chất hình thái sóng khó nhận diện hơn từ ảnh.

So sánh trực tiếp với PULSE gốc (cùng giao thức)

Bài báo gốc PULSE chỉ công bố macro-AUC trên PTB-XL Super gồm 5 siêu lớp (82.4) @pulse , không báo cáo trên tập nhãn chi tiết 44/12/19 mà luận văn đánh giá; do đó để có một so sánh thực sự công bằng với bài gốc, chúng tôi chạy CHÍNH mô hình PULSE-7B gốc (chưa qua QLoRA của luận văn, không gắn adapter) qua đúng giao thức teacher-forced trên cùng tập kiểm thử fold-10 và cùng ba tập nhãn. Bảng dưới đối chiếu PULSE-base với v2.

#figure(
  table(
    columns: 4,
    inset: 4pt,
    align: (left,) + (center,) * 3,
    table.header([*Tác vụ*], [*PULSE-7B base (zero-shot)*], [*v2 (QLoRA fine-tune)*], [*Δ*]),
    [diagnostic (44)], [0.835], [0.893], [+0.058],
    [rhythm (12)], [0.774], [0.887], [+0.113],
    [form (19)], [0.728], [0.849], [+0.121],
    [Trung bình], [0.779], [0.876], [+0.097],
  ),
  caption: [So sánh trực tiếp PULSE-7B gốc (chưa fine-tune) với v2, cùng tập kiểm thử fold-10 và cùng giao thức teacher-forced. Đây là số đo mô hình PULSE trên benchmark của luận văn, không phải con số Super 5-lớp trong giấy gốc.],
)


Hai quan sát đáng chú ý. Thứ nhất, PULSE-base đã đạt macro-AUC trung bình 0.779 dù CHƯA hề được huấn luyện trên định dạng nhãn nhị phân "0"/"1" của luận văn: điều này cho thấy biểu diễn thị giác ECG mà PULSE học sẵn đã mang thông tin phân biệt mạnh, và giao thức teacher-forced khai thác được thông tin đó bằng cách đọc xác suất tại vị trí slot cố định mà không đòi hỏi mô hình tự sinh đúng khối nhãn. Thứ hai, việc tinh chỉnh QLoRA của luận văn mang lại mức tăng nhất quán trên cả ba tác vụ (trung bình +0.097), lớn nhất ở form (+0.121) và rhythm (+0.113) — đúng hai nhóm mã mà phiên bản v1 đánh mất hoàn toàn — và nhỏ hơn ở diagnostic (+0.058) vốn đã được PULSE đặt trọng tâm. Như vậy đóng góp của v2 không chỉ là vá lỗi sụp đổ của v1 mà còn vượt qua chính mô hình nền PULSE trên cùng một phép đo có kiểm soát.

= Thảo luận

Kết quả của phiên bản v2 cho thấy việc chuyển từ thiết kế thử nghiệm ASL-Gen sang công thức nguyên gốc của PULSE đã vá đồng thời cả hai trục hỏng quan sát được ở v1. Ở v1, mô hình sụp đổ về dự đoán toàn norm với macro-AUC xấp xỉ 0.5, đồng thời khoảng 73% mẫu cho khối nhãn hỏng khi sinh tự do nên không trích được điểm. Nguyên nhân gốc là sự lệch kiến trúc: hàm mất mát bất đối xứng được định nghĩa cho head sigmoid với K logit độc lập, mỗi nhãn một xác suất riêng và các siêu tham số gamma âm lớn hơn gamma dương cùng biên m @ridnik , hoàn toàn không phù hợp với một decoder sinh token vốn ràng buộc các token bằng softmax trên toàn bộ từ vựng. Khi áp một loss bất đối xứng lên xác suất token "1"/"0" của một khối 44 nhãn chẩn đoán duy nhất, tín hiệu huấn luyện vừa lệch khỏi cơ chế sinh, vừa cộng hưởng với mất cân bằng nặng (norm chiếm khoảng 44% PTB-XL) @ptbxlimb  và bản chất cross-entropy per-token trên khối đa số "0", dẫn tới mode-collapse và thoái hoá chuỗi sinh @welleck .

Phiên bản v2 giải quyết vấn đề bằng những lựa chọn thiết kế tối giản và không xâm phạm lõi PULSE. Thứ nhất, việc phân rã ba task riêng cho diag (44), rhythm (12) và form (19) khớp với cách benchmark vận hành PTB-XL như các task độc lập @strodthoff  và với định dạng ECGInstruct của PULSE @pulse , cho phép mỗi task có prompt và quy tắc nhãn phù hợp. Thứ hai, quy tắc nhãn được tách theo bản chất dữ liệu: diag dùng ngưỡng likelihood lớn hơn hoặc bằng 50, còn rhythm và form dùng presence vì likelihood bị gán bằng 0 @wagner . Đây chính là chỗ sửa lỗi nuốt form/rhythm của v1, nơi ngưỡng likelihood áp đồng loạt sẽ loại sạch hai nhóm này. Thứ ba, loss trở về cross-entropy next-token nguyên gốc của PULSE, với tuỳ chọn reweight cân bằng lớp @cui  và focal chỉ như ablation @lin ; điều này nhất quán với nhận định rằng focal là loss không proper @mukhoti . Thứ tư, vision encoder được đông cứng, chỉ tinh chỉnh LLM bằng QLoRA 4-bit (r=16, alpha=32), giữ nguyên năng lực biểu diễn ảnh ECG của PULSE.

Một đóng góp về phương pháp đánh giá là quy trình teacher-forced. Thay vì sinh tự do rồi chật vật trích nhãn từ chuỗi có thể hỏng, v2 tự dựng khối nhãn và đọc xác suất P(y\_k=1) bằng softmax trên cặp logit của hai token "0"/"1" tại đúng vị trí đã biết trong một forward pass duy nhất. Cách này đưa tỷ lệ mẫu hỏng về 0% và cho phép tính macro-AUC không phụ thuộc ngưỡng (lấy trung bình trên các lớp có mẫu dương) cùng hiệu chuẩn ngưỡng per-class. Giao thức này không rò rỉ nhãn: ô nhãn luôn điền token "0" cố định độc lập nhãn thật, xác suất đọc tại logit DỰ ĐOÁN ngay trước ô đó nên nhãn thật không vào context. Một nghiên cứu cắt bỏ trên fold-10 củng cố điều này: giữ nguyên khối nhãn nhưng xoá ảnh (ảnh xám) hoặc gán sai ảnh thì macro-AUC sụp từ 0.945 về mức ngẫu nhiên (0.50 và 0.43) — nếu điểm số rút từ nhãn trong khối answer thì hai biến thể này vẫn phải cho AUC cao, nên toàn bộ tín hiệu phân biệt đến từ ảnh chứ không từ nhãn rò rỉ. Một biến thể đổi token điền "0" thành "1" vẫn cho AUC cao hơn hẳn mức ngẫu nhiên (0.80 so với 0.50), xác nhận điểm số là phân phối dự đoán của mô hình chứ không phải sao chép token điền sẵn. Bằng chứng thêm: mean P(1) ở lớp positive luôn lớn hơn rõ rệt so với negative (diag 0.44 so với 0.03, rhythm 0.64 so với 0.06, form 0.35 so với 0.05).

Về vị thế so sánh, trên tập kiểm thử độc lập fold-10, v2 đạt trung bình macro-AUC 0.876 (diag 0.893, rhythm 0.887, form 0.849). So với baseline đa phương thức, con số này vượt xa LLaVA-1.6 backbone zero-shot (50.0) và GPT-4o (55.6), đồng thời vượt nhẹ mức PULSE-7B zero-shot trên PTB-XL Super (82.4) @pulse  và GEM trên cùng phân nhóm Super (83.4) @gem  — và cần lưu ý hai mốc Super chỉ phân loại 5 lớp, trong khi v2 đánh giá trên toàn bộ tập nhãn chi tiết (44+12+19 mã) khó hơn nhiều. Quan trọng hơn, trong so sánh có kiểm soát nhất — cùng tập kiểm thử fold-10 và cùng giao thức teacher-forced — v2 vượt chính mô hình nền PULSE-7B chưa fine-tune (0.779) thêm trung bình +0.097 macro-AUC, với mức tăng lớn nhất ở form và rhythm. So với trần bài toán là CNN 1-D đơn phương thức của Strodthoff (diag 0.937, rhythm 0.957, form 0.896) @strodthoff , vẫn còn khoảng cách, nhưng điều này hợp lý vì baseline làm việc trực tiếp trên tín hiệu 1-D đầy đủ, còn mô hình của luận văn là đa phương thức trên ẢNH ECG qua CLIP ViT-L/14\@336 @pulse  và chỉ tinh chỉnh nhẹ 2 epoch.

= Hạn chế

Thứ nhất, mô hình mới chỉ tinh chỉnh 2 epoch (dừng do giới hạn tài nguyên, không do hội tụ), nên dư địa cải thiện còn lớn và các con số nên được hiểu là cận dưới hơn là hiệu năng bão hoà. Thứ hai, task form yếu nhất (macro-AUC test 0.849), với những lớp như tab chỉ đạt khoảng 0.34; điều này phản ánh cả độ khó nội tại của các phát biểu hình thái lẫn việc bốn mã chồng lấn giữa diag và form @wagner  làm nhiễu ranh giới task. Thứ ba, macro-F1 sau hiệu chuẩn ngưỡng per-class còn thấp (0.22–0.33 trên test) do trung bình đều trên cả các lớp hiếm, vốn là đặc thù phân phối dài đuôi của PTB-XL; vì vậy macro-AUC (đo khả năng xếp hạng) mới là chỉ số headline, và luận văn KHÔNG khẳng định một bộ phân lớp đã hiệu chuẩn sẵn sàng triển khai lâm sàng. Thứ tư, năng lực reasoning chưa được huấn luyện: mô hình hiện chỉ học xuất nhãn, nên đầu ra lập luận của định dạng đa phương thức chưa có giám sát tương ứng và chưa thể đánh giá. Thứ năm, mới đánh giá trên một lần chia fold cố định; cần đánh giá đa fold và hiệu chuẩn xác suất để khẳng định khả năng tổng quát hoá ngoài fold-10.

= Hướng phát triển

Hướng trực tiếp nhất là tăng số epoch huấn luyện để khai thác dư địa còn lại, đặc biệt cho task form và các lớp hiếm, kèm khảo sát kỹ hơn các biến thể reweight cân bằng lớp @cui . Cần đánh giá chéo nhiều fold thay vì một lần chia cố định để kiểm tra dấu hiệu khớp quá và dịch chuyển phân phối giữa các fold, đồng thời hiệu chuẩn xác suất để nâng macro-F1 ở các lớp hiếm. Tiếp theo là bổ sung giám sát reasoning: tận dụng dữ liệu hướng dẫn y khoa đa phương thức theo tinh thần llavamed @llavamed  và các cặp hỏi–đáp ECG dạng ecgqa @ecgqa  để mô hình không chỉ xuất nhãn mà còn sinh lập luận có căn cứ, qua đó tăng tính diễn giải cho chẩn đoán. Cuối cùng, mở rộng so sánh có kiểm soát trên cùng split chuẩn của Strodthoff @strodthoff  với cả baseline đơn phương thức @strodthoff  lẫn các phương pháp đa phương thức @pulse @gem .

= Kết luận

Luận văn đã chỉ ra rằng thất bại của v1 không nằm ở dữ liệu hay backbone mà ở sự lệch giữa một loss bất đối xứng kiểu sigmoid độc lập @ridnik  và cơ chế sinh token của decoder, dẫn tới mode-collapse @welleck  và khối nhãn hỏng. Đóng góp cốt lõi của v2 là một giải pháp tối giản, không sửa lõi PULSE, vá đồng thời cả hai trục collapse và format: phân rã ba task đúng với cấu trúc PTB-XL @strodthoff @pulse , tách quy tắc nhãn để cứu form/rhythm @wagner , trở về cross-entropy next-token nguyên gốc, và chỉ tinh chỉnh LLM bằng QLoRA trên vision đông cứng. Bên cạnh đó, quy trình đánh giá teacher-forced là đóng góp về phương pháp, đưa tỷ lệ mẫu hỏng về 0% và cho phép đo macro-AUC không phụ thuộc ngưỡng trên các lớp có mẫu dương. Kết quả trung bình macro-AUC 0.876 trên tập kiểm thử độc lập fold-10 của ảnh ECG đa phương thức, vượt chính mô hình nền PULSE-7B +0.097 trong so sánh có kiểm soát và vượt xa các backbone tổng quát @pulse @gem , cho thấy một mô hình ngôn ngữ–thị giác có thể phân biệt ECG đa nhãn ở mức cạnh tranh mà không cần đến tín hiệu 1-D — dù macro-F1 còn thấp nên cần củng cố thêm trước khi nói tới độ tin cậy lâm sàng, mở đường cho các bước mở rộng về huấn luyện sâu hơn và bổ sung năng lực lập luận.

#v(0.5em)
#text(size: 9pt)[
  #set par(leading: 0.45em)
  #bibliography("refs.bib", title: "Tài liệu tham khảo", style: "ieee")
]
