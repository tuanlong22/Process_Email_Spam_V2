# Process_email_spam

Version của repo cũ bị lỗi local do file nén tại file lưu log vượt quá 100Mb(phạm vi push 1 lần) nên tôi bỏ repo cũ rồi nha. Mọi update đều ở file này.

**Tên Ứng Dụng: GMail Experience**
![Common](https://github.com/huyvu15/Process_email_spam/blob/main/Image/common.png)

![Have Spam](https://github.com/huyvu15/Process_email_spam/blob/main/Image/Have_Spam.png)


![Client](https://github.com/huyvu15/Process_email_spam/blob/main/Image/Gmai_Client.png)

![Server](https://github.com/huyvu15/Process_email_spam/blob/main/Image/Gmail_server.png)




# Chương trình Gmail Server

Chương trình của bạn có chức năng mô phỏng một máy chủ email đơn giản. Dưới đây là mô tả về cách chương trình hoạt động:

## 1. Giao diện đồ họa

- Chương trình có một giao diện đồ họa sử dụng thư viện Tkinter của Python.

- Giao diện chia thành hai phần chính: "Inbox Frame" và "Email Frame".

- "Inbox Frame" chứa hai danh sách (Listbox) hiển thị tin nhắn "Ham" và "Spam" cùng với nút để xóa tin nhắn đã chọn.

## 2. Gửi và nhận tin nhắn

- Máy chủ mở một socket và lắng nghe kết nối từ các máy khách.

- Khi máy khách kết nối, một luồng mới được tạo để xử lý việc nhận tin nhắn từ máy khách.

- Mỗi khi máy khách gửi tin nhắn, máy chủ nhận và xử lý tin nhắn đó.

- Tin nhắn được phân loại là "Ham" hoặc "Spam" bằng một mô hình Naive Bayes được huấn luyện trước.

## 3. Phân loại tin nhắn

- Khi máy chủ nhận được một tin nhắn mới từ máy khách, nó tạo một DataFrame từ tin nhắn đó và sử dụng mô hình Naive Bayes để phân loại tin nhắn là "Ham" hoặc "Spam".

- Kết quả phân loại được hiển thị trong cửa sổ console và tin nhắn được thêm vào danh sách tương ứng ("Ham" hoặc "Spam").

## 4. Hiển thị tin nhắn

- Khi người dùng chọn một tin nhắn từ danh sách "Ham" hoặc "Spam", nội dung của tin nhắn đó được hiển thị trong khu vực "Email Frame".

- Hình ảnh đại diện của người gửi tin nhắn cũng được hiển thị nếu có.

## 5. Xóa tin nhắn

- Người dùng có thể chọn tin nhắn từ danh sách "Ham" hoặc "Spam" và nhấn nút để xóa tin nhắn đã chọn khỏi danh sách.

## 6. Mô hình Naive Bayes

- Mô hình Naive Bayes được huấn luyện trước với dữ liệu spam từ tệp 'SMSSpamCollection'.

- Khi máy chủ nhận được tin nhắn mới, nó sử dụng mô hình này để dự đoán xem tin nhắn có phải là "Ham" hay "Spam".
