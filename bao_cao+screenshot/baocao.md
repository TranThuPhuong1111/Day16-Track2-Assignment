Báo cáo ngắn: LightGBM trên CPU (AWS EC2 t3.small, 2 vCPU / 2 GB RAM)

Mô hình LightGBM phát hiện gian lận thẻ tín dụng, huấn luyện trên dataset creditcardfraud (284.807 dòng, tỷ lệ gian lận ~0,17%), chạy hoàn toàn trên CPU.

Thời gian: load dữ liệu mất 2,25 s. Training mất 28,2 s cho 715 vòng boosting. Với 227 nghìn dòng và 30 đặc trưng, con số này cho thấy LightGBM đủ nhanh trên CPU nhỏ mà không cần GPU.
Chất lượng mô hình: AUC-ROC đạt 0,977. Accuracy 0,9996 nhưng chỉ số này ít ý nghĩa vì dữ liệu cực kỳ mất cân bằng. Precision 0,95, Recall 0,79 và F1 0,86 phản ánh đúng hơn: mô hình bắt được khoảng 79% giao dịch gian lận và hầu như không báo nhầm.
Inference: dự đoán 1 dòng mất khoảng 1,28 ms (phần lớn là overhead gọi hàm). Dự đoán theo lô 1000 dòng mất khoảng 70,7 ms, tức khoảng 14.000 dòng/giây, đủ cho nhiều tác vụ phát hiện gian lận gần thời gian thực.
Tài nguyên: khi chạy, python3 chiếm khoảng 107% CPU và dùng dưới 400 MiB RAM, nên máy nhỏ vẫn đủ.
Kết luận: với dữ liệu dạng bảng, mô hình gradient boosting như LightGBM chạy tốt trên CPU, chi phí thấp. GPU chỉ thực sự cần cho các mô hình lớn như LLM. Chi phí chủ yếu đến từ NAT Gateway và ALB, không phải từ instance.