# Mage terminal agent cho Windows

Sử dụng model `microsoft/Mage-VL-Cowork` đã tải trong Hugging Face cache. Model chạy tại máy, mặc định NF4 4-bit trên CUDA. Không cần API key hay tải lại weights.

Mở PowerShell trong thư mục này:

```powershell
.\mage.cmd
```

Nhập yêu cầu tại `Task>`. Model được giữ trong bộ nhớ giữa các task. Ví dụ: `Mở Notepad và gõ Xin chào Mage`.

Chạy một task trực tiếp:

```powershell
.\mage.cmd "Mở Notepad và gõ Xin chào Mage"
```

Giữ F8 để dừng trước thao tác tiếp theo, hoặc Ctrl+C trong terminal. Nhập `/exit` để thoát. Trong lúc suy luận, F8 được kiểm tra trước khi thực thi kết quả. Giữ màn hình mở và tránh dùng chuột, bàn phím khi agent đang làm việc. Agent điều khiển màn hình chính và có thể chạy lệnh PowerShell với quyền của tiến trình hiện tại. Công cụ tên `bash` được ánh xạ sang PowerShell trên Windows; phiên shell không được giữ giữa các lần gọi.

Các tùy chọn:

```powershell
.\mage.cmd --check
.\mage.cmd --gui-only
.\mage.cmd --max-steps 50 --max-tokens 2048
.\mage.cmd --model "D:\models\Mage-VL-Cowork"
.\mage.cmd --cpu
```

`--check` kiểm tra nạp model và suy luận với ảnh đỏ tổng hợp, không thao tác desktop. `--gui-only` chỉ cho phép chuột/bàn phím. CPU dùng float32, cần nhiều RAM và chạy chậm. Mặc định giữ một ảnh màn hình gần nhất; đạt giới hạn bước sẽ báo chưa hoàn tất.

Mỗi task tự ghi log tại `.logs/<thời điểm>.jsonl`, bao gồm yêu cầu, phản hồi model, thời gian suy luận, số token, lệnh đã chạy và kết quả. Terminal in đường dẫn log ngay khi bắt đầu task. Mặc định cho phép 4096 token mỗi lượt để tránh cắt mất lệnh ở cuối phản hồi. Khi model trả định dạng lỗi, bộ chạy yêu cầu trả lại lệnh hoàn chỉnh trong giới hạn số bước.

`mage.cmd` ưu tiên môi trường `.venv` riêng của dự án, đã cài Transformers 4.57.3, Hugging Face Hub 0.36.0 và Tokenizers 0.22.1 theo `requirements-runtime.txt`. Torch, accelerate, bitsandbytes, Pillow và PySide6 dùng bản đã có trên máy. Thư viện Python hệ thống được giữ nguyên. Mã kiến trúc Python của model được chạy bằng `trust_remote_code=True` từ bản cache cục bộ; cache module phát sinh nằm trong `.cache/modules` của dự án.

Phần ngôn ngữ dùng NF4 4-bit; phần nhìn và đầu ra giữ độ chính xác gốc. Bộ nạp ánh xạ tên weights của checkpoint sang `model.language_model` cho mã kiến trúc hiện có, kiểm tra thiếu/thừa weights và bổ sung vị trí patch ảnh. `mage_system_prompt.txt` lấy từ model card chính thức của Microsoft (Apache-2.0), có phần bổ sung để dùng PowerShell trên Windows. Tùy chọn `--no-thinking` mang tính thử nghiệm.

Bộ tương thích sửa vị trí token khi giải mã có cache trên Transformers 5, và chuyển phím mở ứng dụng Linux `Alt+F2` mà model có thể sinh ra sang hộp Run của Windows (`Win+R`). Phản hồi công cụ có nhãn `tool_response`, tên ứng dụng đang được focus và ảnh desktop mới, kể cả sau lệnh PowerShell. Ảnh giữ độ phân giải tối đa 1920×1200, mặc định giữ một ảnh gần nhất để phù hợp GPU 6 GB; tùy chọn `--images-to-keep` thay đổi số ảnh. Nhập Unicode được giãn 25 ms mỗi mã UTF-16 để tránh Notepad WinUI lặp ký tự cuối. Kiểm tra hồi quy bằng `.\.venv\Scripts\python.exe -m unittest test_generation_compat -v`.

Kiểm tra điều khiển desktop riêng bằng `python verify_desktop.py`; phép thử tạo và đóng một cửa sổ nhập liệu tạm.

Chỉ lệnh có JSON hợp lệ mới được chạy. Với yêu cầu dạng `Mở Notepad và gõ <văn bản>` hoặc `Open Notepad and type <text>`, bộ chạy xác minh trực tiếp nội dung RichEdit đang hiển thị trong ứng dụng Notepad đang focus rồi dừng, tránh model tiếp tục lưu/đóng ngoài yêu cầu. Các dạng task khác dùng lệnh `terminate` của model. Kiểm tra nhập liệu thực tế bằng `python verify_notepad.py` sau task mẫu sẽ đọc điều khiển RichEdit trong các cửa sổ Notepad đang hiển thị và yêu cầu đúng `Xin chào Mage`; ô tên file và các tab không hiển thị không được tính.

Tài liệu model: https://huggingface.co/microsoft/Mage-VL-Cowork
