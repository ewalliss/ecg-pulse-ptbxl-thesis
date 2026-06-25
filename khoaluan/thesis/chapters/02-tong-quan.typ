= Cơ sở lý thuyết và tổng quan

== Cơ sở lý thuyết y sinh và hệ thống điện tâm đồ

=== Giải phẫu cơ học và hệ thống dẫn truyền điện học của tim

Để xây dựng một hệ thống nhận dạng tự động các bất thường về tim mạch từ dữ liệu đa phương thức, việc thấu hiểu bản chất cơ học và động lực học điện thế của tim là điều kiện tiên quyết. Về mặt giải phẫu học, tim con người là một cơ quan cơ rỗng đóng vai trò như một máy bơm kép được chia thành bốn buồng chức năng độc lập: hai tâm nhĩ (atria) nằm ở phía trên đảm nhận nhiệm vụ tiếp nhận máu trở về tim, và hai tâm thất (ventricles) nằm ở phía dưới chịu trách nhiệm co bóp mạnh để tống máu vào hệ thống tuần hoàn đại tuần hoàn và tiểu tuần hoàn. Do phải chịu áp lực cơ học rất lớn để đẩy máu đi nuôi toàn bộ cơ thể, thành cơ của tâm thất — đặc biệt là tâm thất trái — có độ dày và khối lượng cơ học vượt trội hơn hẳn so với thành cơ tâm nhĩ.

#figure(
  image("../cosolythuyet/images/heart_structure.png", width: 58%),
  caption: [Hình 2.1: Cấu trúc bốn buồng của tim — hai tâm nhĩ (atria) ở trên, hai tâm thất (ventricles) ở dưới, cùng các van tim và mạch máu lớn.],
  kind: image,
) <fig-heart-structure>

Hoạt động co bóp cơ học đồng bộ này không tự diễn ra mà được điều khiển một cách nghiêm ngặt bởi một hệ thống các tế bào cơ tim chuyên biệt có khả năng tự phát bản tin điện học (Hệ thống dẫn truyền điện tim). Quy trình kích hoạt điện học trong một chu chuyển tim tiêu chuẩn được diễn ra theo một trình tự tuyến tính nghiêm ngặt như sau:

+ *Khởi cực tại Nút xoang nhĩ (Sinoatrial - SA Node):* Nằm ở tâm nhĩ phải, nút SA đóng vai trò là trung tâm phát nhịp tự nhiên của tim, tự động phát ra các xung điện sinh học với tần số chuẩn từ 60 đến 100 lần mỗi phút.
+ *Lan truyền cơ tim nhĩ:* Xung điện từ nút SA lan tỏa nhanh chóng qua các thớ cơ của cả hai tâm nhĩ, kích thích quá trình khử cực và ép hai tâm nhĩ co bóp để đẩy máu xuống tâm thất.
+ *Chuyển tiếp tại Nút nhĩ thất (Atrioventricular - AV Node):* Tín hiệu điện hội tụ tại nút AV. Tại đây, cấu trúc mô học đặc thù tạo ra một sự chậm trễ sinh lý nhằm đảm bảo tâm nhĩ đã tống toàn bộ lượng máu xuống thất trước khi tâm thất bắt đầu co bóp; độ trễ này là thành phần chính của khoảng PR (toàn khoảng PR kéo dài 0.12 đến 0.20 giây).
+ *Hệ thống bó His và mạng Purkinje:* Sau khi đi qua nút AV, xung điện sinh học di chuyển tốc độ cao dọc theo bó His, phân tách thành hai nhánh bó chính: nhánh bó phải (Right Bundle Branch - RBB) và nhánh bó trái (Left Bundle Branch - LBB), chạy sâu vào vách liên thất và tận cùng bằng mạng lưới Purkinje. Mạng lưới này phân phối dòng điện đồng đều đến từng tế bào cơ tâm thất, kích hoạt sự khử cực đồng bộ của khối cơ thất.

#figure(
  image("../cosolythuyet/images/conduction.png", width: 58%),
  caption: [Hình 2.2: Hệ thống dẫn truyền điện học của tim — nút SA, nút AV, bó His, nhánh trái (LBB) và nhánh phải (RBB), mạng Purkinje.],
  kind: image,
) <fig-heart-conduction>

=== Hệ thống 12 đạo trình lâm sàng và cấu trúc chuỗi sóng P-Q-R-S-T

Điện tâm đồ (Electrocardiogram - ECG) là đồ thị biểu diễn sự biến thiên của các vector tổng điện thế sinh học do dòng điện tim tạo ra theo thời gian. Các tín hiệu này được ghi nhận thông qua các điện cực tiếp xúc đặt tại các vị trí giải phẫu cố định trên cơ thể bệnh nhân. Trong quy trình lâm sàng tiêu chuẩn, một bản ghi hoàn chỉnh bao gồm 12 đạo trình (12-Lead ECG), chia không gian quan sát quả tim thành hai mặt phẳng toán học độc lập:

- *Nhóm chuyển đạo chi (Mặt phẳng trán):* Bao gồm 3 đạo trình chuẩn lưỡng cực (I, II, III) và 3 đạo trình đơn cực tăng cường (aVR, aVL, aVF). Nhóm này quan sát dòng điện tim theo chiều dọc (trên - dưới, trái - phải). Trong đó, các đạo trình vùng hoành (II, III, aVF) nhìn trực diện vào phần dưới và sau của tâm thất.
- *Nhóm chuyển đạo trước tim (Mặt phẳng ngang):* Bao gồm 6 đạo trình đơn cực ký hiệu từ V_1 đến V_6 đặt tuần tự trên thành ngực của bệnh nhân. Nhóm này khu trú không gian theo chiều ngang nhằm quan sát từng vùng cơ tim cụ thể: V_1, V_2 đối diện vách liên thất; V_3, V_4 quan sát vùng thành trước tim; và V_5, V_6 quan sát vùng thành bên của tâm thất trái.

#figure(
  image("../cosolythuyet/images/leads.png", width: 80%),
  caption: [Hình 2.3: Vị trí đặt các điện cực trên cơ thể — 4 điện cực chi (RA, LA, RL, LL) và 6 điện cực trước tim (V1–V6); hình bên phải minh họa V5, V6 theo đường nách trước/giữa.],
  kind: image,
) <fig-leads-placement>

#figure(
  image("../cosolythuyet/images/ecg_paper.png", width: 78%),
  caption: [Hình 2.4: Một bản ghi điện tâm đồ 12 đạo trình thực tế trên giấy kẻ ô tiêu chuẩn.],
  kind: image,
) <fig-ecg-paper>

Khi dòng điện sinh học lan truyền qua hệ thống đạo trình này, nó vẽ nên một chuỗi các xung điện có tính chu kỳ trên giấy in ECG. Một chu chuyển tim tiêu chuẩn (một nhịp đập) được đặc trưng hình học bởi chuỗi sóng P-Q-R-S-T và các khoảng thời gian chuyển tiếp lâm sàng sau đây:

+ *Sóng P (P wave) – Khử cực tâm nhĩ*
  - *Vị trí giải phẫu:* Là làn sóng nhỏ đầu tiên xuất hiện trong một chu kỳ tim, đứng ngay trước phức bộ QRS. Sóng P phản ánh quá trình xung điện xuất phát từ nút SA lan tỏa làm khử cực điện học và kích hoạt co bóp cơ học của hai tâm nhĩ.
  - *Đặc điểm hình học:* Sóng P tiêu chuẩn có dạng vòm tròn đối xứng, mịn, không có khía răng cưa. Biên độ (độ cao) bình thường không được vượt quá 2.5 "mm" (0.25 "mV") và thời gian kéo dài (độ rộng) dao động từ 0.08 đến 0.11 giây. Mọi biến đổi làm sóng P cao nhọn hoặc giãn rộng đều phản ánh các hội chứng phì đại buồng tâm nhĩ.

+ *Khoảng PR (PR Interval) – Thời gian dẫn truyền nhĩ thất*
  - *Vị trí giải phẫu:* Được tính từ thời điểm bắt đầu sóng P cho đến thời điểm bắt đầu phức bộ QRS. Khoảng thời gian này đại diện cho tổng thời gian dòng điện di chuyển từ tâm nhĩ, đi qua nút nhĩ thất (AV Node) để chuẩn bị kích hoạt tâm thất.
  - *Đặc điểm hình học:* Thời gian tiêu chuẩn kéo dài từ 0.12 đến 0.20 giây (tương ứng với 3 đến 5 ô nhỏ trên giấy đồ thị ECG tiêu chuẩn). Nếu khoảng PR kéo dài vượt ngưỡng 0.20 giây, hệ thống dẫn truyền đang bị nghẽn sinh lý (Hội chứng Block nhĩ thất).

+ *Phức bộ QRS (QRS Complex) – Khử cực tâm thất*
  Đây là thành phần quan trọng nhất, sở hữu biên độ lớn nhất và cấu trúc hình học phức tạp nhất trên đồ thị ECG, thể hiện quá trình khử cực mạnh mẽ của khối cơ hai tâm thất. Do khối lượng cơ tâm thất rất lớn, phức bộ này luôn có năng lượng tín hiệu vượt trội, bao gồm ba sóng thành phần:
  - *Sóng Q (Q wave):* Sóng âm (đi xuống dưới đường đẳng điện) đầu tiên của phức bộ, xuất hiện ngay trước sóng R. Sóng Q bình thường rất nhỏ hoặc không xuất hiện ở một số đạo trình. Nếu sóng Q sâu quá 1/4 chiều cao sóng R và rộng hơn 0.04 giây, nó trở thành Sóng Q bệnh lý (Pathologic Q-wave), minh chứng cho một vùng cơ tim đã bị hoại tử hoàn toàn do Nhồi máu cơ tim (MI).
  - *Sóng R (R wave):* Sóng dương (đi lên trên đường đẳng điện) đầu tiên và thường là đỉnh cao nhất của chu kỳ tim. Chiều cao của sóng R phản ánh trực tiếp khối lượng cơ học của tâm thất; sóng R vọt cao bất thường ở các đạo trình trước tim trái (V_5, V_6) là dấu hiệu cốt lõi của hội chứng Phì đại thất trái (LVH).
  - *Sóng S (S wave):* Sóng âm đi ngay sau sóng R dương.
  - *Đặc điểm hình học:* Độ rộng thời gian của toàn bộ phức bộ QRS bình thường cực kỳ ngắn, chỉ dao động từ 0.06 đến 0.10 giây. Nếu phức bộ này giãn rộng vượt ngưỡng 0.12 giây, hệ thống dẫn truyền ở hai nhánh bó His đã bị tổn thương, khiến hai tâm thất khử cực lệch pha, gây ra hình ảnh đỉnh đôi chữ M (dạng sóng R S R') của hội chứng Block nhánh (LBBB/RBBB).

+ *Đoạn ST (ST Segment) – Giai đoạn tái cực sớm của thất*
  - *Vị trí giải phẫu:* Bắt đầu từ điểm J (điểm kết thúc của phức bộ QRS) đến điểm bắt đầu của sóng T. Đây là giai đoạn hoàn toàn tâm thất đã khử cực xong và chuẩn bị bước vào pha hồi phục điện thế.
  - *Đặc điểm hình học:* Ở trạng thái cơ tim lành lặn, đoạn ST bắt buộc phải nằm phẳng hoàn toàn trên đường đẳng điện (baseline). Sự thay đổi đường nét của đoạn ST như ST chênh lên (ST elevation) dạng vòm hoặc ST chênh xuống (ST depression) dạng đi ngang/dốc xuống là bằng chứng vàng để nhận diện các tổn thương thiếu máu cục bộ cơ tim cấp tính hoặc tổn thương xuyên thành.

+ *Sóng T (T wave) – Tái cực tâm thất*
  - *Vị trí giải phẫu:* Làn sóng rộng xuất hiện ngay sau đoạn ST, thể hiện quá trình tái cực phục hồi trạng thái điện thế màng ban đầu của cơ tâm thất.
  - *Đặc điểm hình học:* Sóng T tiêu chuẩn có hình dáng bất đối xứng (sườn lên dốc thoải, sườn xuống dốc đứng). Sóng T bị đảo ngược (trở thành sóng âm) hoặc cao nhọn đối xứng (sóng T khổng lồ) phản ánh tình trạng thiếu máu cơ tim hoặc rối loạn nồng độ kali máu nghiêm trọng.

#figure(
  image("../cosolythuyet/images/pqrst.png", width: 92%),
  caption: [Hình 2.5: Cấu trúc hình học của một chu kỳ sóng P-Q-R-S-T và các khoảng thời gian lâm sàng (khoảng PR, khoảng QT, đoạn TP).],
  kind: image,
) <fig-pqrst-complex>

== Phân loại các hội chứng bệnh lý mục tiêu dựa trên điện tâm đồ

Dựa trên tiêu chuẩn y tế quốc tế SCP-ECG của tập dữ liệu PTB-XL, hệ thống phân loại được cấu trúc thành ba miền chức năng độc lập: chẩn đoán bệnh lý thực thể (Diagnostic), nhận diện rối loạn nhịp (Rhythm), và phân tích biến đổi hình thái (Form). Việc phân rã này giúp ánh xạ chính xác bản chất y sinh của từng nhóm bệnh lên cấu trúc hình học của đồ thị ECG, đồng thời là căn cứ cho cách khóa luận tách bài toán PTB-XL thành ba tác vụ độc lập (Chương 3).

=== Nhóm chẩn đoán bệnh lý thực thể (Diagnostic)

Nhóm chẩn đoán tập trung vào việc phát hiện các tổn thương cấu trúc vĩ mô của cơ tim, các hội chứng suy giảm chức năng huyết động dài hạn, và các rối loạn dẫn truyền vùng lớn. Theo chuẩn SCP-ECG của PTB-XL, nhóm này gồm các siêu lớp NORM (bình thường), MI (nhồi máu), STTC (biến đổi ST/T), HYP (phì đại) và CD (rối loạn dẫn truyền). Ba đại diện lâm sàng tiêu biểu bao gồm:

+ *Nhồi máu cơ tim (Myocardial Infarction - MI):* Đây là tình trạng một vùng cơ tim bị thiếu máu cục bộ hoại tử hoàn toàn do tắc nghẽn động mạch vành cấp tính. Về mặt điện sinh học, vùng cơ tim bị tổn thương hoặc hoại tử sẽ mất khả năng phân cực và dẫn truyền xung điện bình thường, tạo ra các "vùng câm điện học".
  - *Biến đổi đồ thị:* Sự mất cân bằng vector điện thế này trực tiếp kéo lệch đoạn ST ra khỏi đường đẳng điện, gây ra hiện tượng ST chênh lên (ST elevation) dạng vòm trong nhồi máu cấp tính xuyên thành, hoặc ST chênh xuống (ST depression) trong tổn thương dưới nội tâm mạc. Khi tế bào cơ tim hoại tử hoàn toàn, tại các đạo trình đối diện trực tiếp vùng tổn thương sẽ xuất hiện các sóng Q bệnh lý (Pathologic Q-wave) sâu vượt quá 1/4 biên độ sóng R và giãn rộng hơn 0.04 giây.

+ *Phì đại cơ tim (Hypertrophy - HYP):* Hội chứng này xảy ra khi các buồng tim (đặc biệt là tâm thất trái - LVH) phải tăng sinh khối lượng và độ dày thành cơ để chống lại áp lực dòng máu mãn tính (như trong bệnh cao huyết áp).
  - *Biến đổi đồ thị:* Sự gia tăng đột biến về khối lượng cơ học làm tăng tổng lượng vector điện thế khử cực của tâm thất. Hệ quả trực tiếp là *biên độ kích thước của phức bộ QRS vọt lên rất cao* so với bình thường. Cụ thể, mô hình thị giác sẽ ghi nhận sóng R tăng vọt độ cao tại các đạo trình trước tim trái (V_5, V_6) và sóng S lún sâu bất thường tại các đạo trình trước tim phải (V_1, V_2).

+ *Rối loạn dẫn truyền — Block nhánh (Conduction Disturbance, vd CLBBB/CRBBB):* Hội chứng này xảy ra khi một trong hai nhánh chính của bó His (nhánh trái hoặc nhánh phải) bị nghẽn hoàn toàn do tổn thương mô học, khiến xung khử cực không lan truyền đồng thời đến hai tâm thất; tâm thất thuộc nhánh lành khử cực trước, rồi tín hiệu phải đi xuyên vách liên thất sang khử cực tâm thất thuộc nhánh tổn thương.
  - *Biến đổi đồ thị:* Do hai tâm thất khử cực lệch pha, tổng thời gian khử cực thất kéo dài, làm *phức bộ QRS giãn rộng* vượt 0.12 giây; tại đạo trình nhìn trực diện vùng tổn thương ($V_1$ với RBBB, $V_6$ với LBBB) xuất hiện hình ảnh *đỉnh đôi chữ M* (dạng $R S R'$). Trong chuẩn SCP-ECG của PTB-XL, các block nhánh hoàn toàn được xếp vào nhóm chẩn đoán (siêu lớp CD), không phải nhóm hình thái.

=== Nhóm rối loạn nhịp (Rhythm)

Nhóm này kiểm soát các bất thường liên quan đến tần số phát xung và tính chu kỳ thời gian của dòng điện tim, phản ánh sự suy giảm chức năng điều khiển của nút xoang (SA Node).

+ *Rung nhĩ (Atrial Fibrillation - AFIB):* Là một trong những dạng rối loạn nhịp nguy hiểm nhất, xảy ra khi nút SA mất quyền kiểm soát hoàn toàn. Lúc này, hàng loạt các ổ phát nhịp hỗn loạn ở tâm nhĩ tự động phát xung với tần số cực cao (350 đến 600 lần/phút), khiến tâm nhĩ không thể co bóp đồng bộ mà chỉ rung lên lăn tăn.
  - *Biến đổi đồ thị:* Xung điện nhĩ hỗn loạn làm *mất hoàn toàn sóng P* phẳng lặng tiêu chuẩn trên toàn bộ 12 đạo trình, thay thế bằng các sóng f (fibrillation waves) lăn tăn vô định hình. Đồng thời, do nút nhĩ thất (AV Node) lọc và truyền xung điện xuống thất một cách ngẫu nhiên, khoảng cách giữa các đỉnh sóng R (khoảng R-R Intervals) trở nên *hoàn toàn bất quy tắc và biến thiên liên tục* trên trục thời gian chuỗi.

+ *Nhịp nhanh xoang (STACH) và Nhịp chậm xoang (SBRAD):* Xảy ra khi trung tâm phát nhịp xoang hoạt động quá mức (> 100 nhịp/phút) hoặc quá thưa thớt (< 60 nhịp/phút) do biến đổi sinh lý hoặc cường thần kinh thực vật.
  - *Biến đổi đồ thị:* Hình thái hình học của từng tổ hợp sóng đơn lẻ (P-QRS-T) hoàn toàn không bị biến dạng. Tuy nhiên, mật độ xuất hiện của các chu kỳ tim trên trục hoành giấy in thay đổi rõ rệt: đoạn TP (thời gian nghỉ giữa hai nhịp tim) bị co ngắn tối đa trong nhịp nhanh và giãn dài ra trong nhịp chậm.

=== Nhóm biến đổi hình thái (Form)

Khác với hai nhóm trên, nhóm hình thái tập trung vào các vi biến đổi hình học khu trú trên từng tổ hợp sóng đơn lẻ, phản ánh tình trạng tổn thương cục bộ cấu trúc mô học cơ tim hoặc rối loạn tái cực khu trú. Theo chuẩn SCP-ECG của PTB-XL, nhóm này gồm các mã như sóng Q bất thường (`QWAVE`), biến đổi ST/T khu trú (`NDT`, `NST_`, `STD_`, `STE_`, `INVT`), điện thế thấp/cao (`LVOLT`, `HVOLT`) hay ngoại tâm thu (`PVC`, `PAC`). Đây là các đặc trưng hình học có mật độ xuất hiện thưa thớt (sparse features) và dễ bị nhiễu nhất trên ảnh đồ thị. (Các block nhánh hoàn toàn không thuộc nhóm này mà thuộc nhóm chẩn đoán — xem Mục 2.2.1.)

Nhóm Form chịu trách nhiệm bóc tách và phân lớp các sai lệch cấu trúc vi mô xuất hiện trên từng phân đoạn sóng cụ thể: sóng Q sâu (nhưng chưa đạt ngưỡng hoại tử của nhóm Diagnostic), biến dạng răng cưa hoặc khía mờ ở đỉnh sóng R, đoạn ST dốc nhẹ bất thường, hay sóng T suy hao biên độ khu trú tại một vài đạo trình riêng lẻ. Các vi biến đổi này không đại diện cho một bệnh lý thực thể diện rộng ngay lập tức, nhưng phản ánh tình trạng xơ hóa cơ tim giai đoạn sớm hoặc rối loạn tái cực cục bộ.

== Bộ dữ liệu PTB-XL và mô hình nền PULSE-7B

=== Tín hiệu ECG và bộ dữ liệu PTB-XL

PTB-XL gán cho mỗi ECG nhiều phát biểu SCP với tổng 71 mã, chia thành 44 mã chẩn đoán, 12 mã nhịp và 19 mã hình thái, trong đó 4 mã chồng lấn giữa chẩn đoán và hình thái @wagner. Một đặc thù quan trọng là giá trị likelihood (0–100) chỉ có ý nghĩa cho nhóm chẩn đoán; nhóm nhịp và hình thái được gán likelihood bằng 0, tức chỉ thể hiện sự hiện diện @wagner. Benchmark của Strodthoff vận hành PTB-XL như nhiều tác vụ riêng và quy định split chuẩn 8 fold huấn luyện / fold 9 kiểm định / fold 10 kiểm thử @strodthoff.

=== Kiến trúc LLaVA và mô hình nền PULSE-7B

PULSE-7B kế thừa kiến trúc họ LLaVA-v1.6-Vicuna-7B, gồm bộ mã hóa thị giác CLIP ViT-L/14 ở độ phân giải 336, một lớp chiếu (projector) căn chỉnh đặc trưng ảnh với không gian ngôn ngữ, và mô hình ngôn ngữ Vicuna-7B @pulse. PULSE được huấn luyện bằng entropy chéo next-token tiêu chuẩn của decoder trên tập chỉ dẫn ECGInstruct, vốn đã phân rã bài toán thành các tác vụ chỉ dẫn riêng @pulse.

Hình 2.6 tóm lược quy trình PULSE xây dựng tập chỉ dẫn ECGInstruct, gồm hai khâu: (1) tổng hợp ảnh ECG từ tín hiệu kèm nhiều biến dạng mô phỏng ảnh chụp thực tế (xoay, nhiễu, đổi bố cục, nếp gấp); và (2) dựng đa tác vụ dưới hướng dẫn của chuyên gia lâm sàng (bốn nhóm tác vụ, bốn kiểu câu hỏi), sinh cặp chỉ dẫn–trả lời bằng Llama 3 rồi tinh chỉnh prompt và kiểm soát chất lượng. Việc PULSE phân rã bài toán theo tác vụ ngay từ khâu dữ liệu là một trong những căn cứ để khóa luận cũng phân rã PTB-XL thành ba tác vụ độc lập (Chương 3).

#figure(
  image("../Hinh_2_1_ecginstruct.png", width: 100%),
  caption: [Hình 2.6: Quy trình xây dựng tập chỉ dẫn ECGInstruct của PULSE (vẽ lại theo @pulse) — tổng hợp ảnh ECG đa biến dạng và dựng đa tác vụ với chuyên gia lâm sàng.],
  kind: image,
)

=== Học sâu đa phương thức cho ECG

Bài toán của khóa luận là đa phương thức: đầu vào gồm ảnh ECG (qua CLIP) và văn bản câu hỏi (qua LLM), đầu ra là văn bản nhãn. Điều này phân biệt với các baseline đơn phương thức chỉ xử lý tín hiệu một chiều như mạng nơ-ron tích chập của Strodthoff @strodthoff.

== Các công trình nghiên cứu liên quan

=== Benchmark và quy ước nhãn trên PTB-XL

Strodthoff và cộng sự thiết lập benchmark học sâu trên PTB-XL với mạng tích chập một chiều, đạt mức trần tham chiếu macro-AUC cao (chẩn đoán 0.937, nhịp 0.957, hình thái 0.896) @strodthoff. Wagner và cộng sự mô tả cấu trúc nhãn SCP và quy ước likelihood của bộ dữ liệu @wagner.

=== Các mô hình ngôn ngữ–thị giác và đa phương thức cho ECG

PULSE dạy LLaVA đọc hiểu ảnh ECG và đạt macro-AUC 82.4 trên PTB-XL Super (5 lớp), vượt xa GPT-4o (55.6) và backbone LLaVA-1.6 chưa tinh chỉnh (50.0) @pulse. PULSE đánh giá trên bộ ECGBench gồm nhiều tác vụ (Hình 2.7): nhóm tái mục đích từ chẩn đoán/báo cáo (Abnormality Detection, Report Generation) và nhóm tạo mới từ tài nguyên ngoài (MMMU-ECG, ECG Arena). Trong đó, tác vụ Abnormality Detection — yêu cầu xác định các nhãn bất thường từ một danh sách ứng viên cho mỗi ảnh ECG — chính là tác vụ mà khóa luận kế thừa và tái cấu trúc cho PTB-XL (Chương 3). GEM kết hợp cả ảnh, tín hiệu và văn bản, đạt 83.4 trên cùng phân nhóm Super @gem.

#figure(
  image("../Hinh_2_2_ecgbench.png", width: 100%),
  caption: [Hình 2.7: Phân loại tác vụ trong bộ đánh giá ECGBench của PULSE (vẽ lại theo @pulse). Khóa luận kế thừa _cách hình thức hóa_ tác vụ Abnormality Detection (ô được làm nổi), nhưng áp trên PTB-XL với ảnh 12 chuyển đạo không biến dạng và đo bằng giao thức teacher-forced (đọc xác suất nhị phân theo từng mã) thay cho sinh tự do — ví dụ minh họa trong ô (ảnh CPSC 12×1 có biến dạng, đáp án sinh tự do) là của PULSE, không phải thiết lập của khóa luận.],
  kind: image,
) ECG-Chat đi theo hướng hội thoại và sinh báo cáo @ecgchat; MERL học biểu diễn tương phản tín hiệu–báo cáo phục vụ phân loại zero-shot @merl; HeartLang xây mô hình nền tự giám sát cho tín hiệu ECG @heartlang; Tang và cộng sự tập trung hỏi–đáp ECG few-shot @tang; Nandakishor tinh chỉnh LoRA trên LLaMA đa phương thức cho ảnh ECG @nandakishor; ECG-QA định dạng ECG thành bài toán hỏi–đáp @ecgqa. Điểm chung là các hệ sinh-ngôn ngữ đều dùng entropy chéo next-token, không dùng một hàm mất mát kiểu phân loại đa nhãn rời rạc trên cơ chế sinh.

=== Hàm mất mát cho dữ liệu mất cân bằng

Để xử lý mất cân bằng lớp, tái trọng số cân bằng theo lớp dựa trên số mẫu hiệu dụng là một lựa chọn proper @cui; trong khi đó focal loss tuy phổ biến nhưng là hàm không proper, có thể làm sai lệch hiệu chuẩn xác suất @lin @mukhoti. Khía cạnh thoái hóa và lặp lại của giải mã sinh khi tín hiệu giám sát lệch cũng đã được phân tích trong y văn @welleck.

#figure(
  table(
    columns: 5,
    inset: 6pt,
    align: left,
    table.header([*Mô hình*], [*Phương thức*], [*Mục tiêu huấn luyện*], [*Định nghĩa nhãn*], [*Lập luận text*]),
    [Strodthoff (CNN 1-D) @strodthoff], [tín hiệu], [BCE đa nhãn, head sigmoid], [đa nhãn theo tác vụ], [không],
    [PULSE @pulse], [ảnh + text], [CE next-token], [chỉ dẫn ECGInstruct], [có],
    [GEM @gem], [ảnh + tín hiệu + text], [CE next-token (sinh)], [chỉ dẫn đa phương thức], [có],
    [ECG-Chat @ecgchat], [tín hiệu + text], [CE next-token (hội thoại)], [hỏi-đáp / báo cáo], [có],
    [MERL @merl], [tín hiệu + text], [tương phản tín hiệu-text], [truy vấn nhãn zero-shot], [hạn chế],
    [HeartLang @heartlang], [tín hiệu], [tự giám sát + tinh chỉnh], [nhãn tác vụ hạ nguồn], [không],
    [Nandakishor @nandakishor], [ảnh/text (LLM)], [CE next-token], [sinh văn bản chẩn đoán], [có],
    [ECG-QA @ecgqa], [tín hiệu + text], [CE next-token], [hỏi-đáp], [có],
    [Khóa luận (PULSE-7B + QLoRA)], [ảnh + text], [CE next-token, prompt phân rã tác vụ], [likelihood (chẩn đoán) / hiện diện (nhịp, hình thái)], [có (để ngỏ)],
  ),
  caption: [Bảng 2.1: So sánh tóm tắt một số mô hình ECG đa phương thức và benchmark liên quan.],
  kind: table,
)

== Đóng góp của khóa luận

Đóng góp của khóa luận không nằm ở một hàm mất mát mới mà ở việc tái cấu trúc bài toán cho khớp với cả quy ước PTB-XL lẫn cơ chế sinh của PULSE: phân rã ba tác vụ, quy tắc gán nhãn hai tầng, trở về entropy chéo next-token nguyên gốc kèm tùy chọn tái trọng số cân bằng lớp, và một giao thức đánh giá teacher-forced. Ngoài ra, khóa luận thiết lập một so sánh có kiểm soát với chính mô hình nền PULSE-7B để tách bạch đóng góp của bước tinh chỉnh.
