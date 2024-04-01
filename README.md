* Cài đặt môi trường ảo
python -m ven venv
* Kích hoạt môi trường ảo
 .\venv\Scripts\activate
* Tải thư viện yêu cầu
 pip install -r requirements.txt
 pip install -r req.txt 

 ----------------------------------Mô tả-------------------------------------------

 backend: Thư mục chứa xử lý
 document_loader: Load các tập dữ liệu như csv,txt, docx, pdf, ppt
 embeddings: Thư mục chứa các file dùng để nhúng dữ liệu(FAISS hoặc Chroma)

 test: Chứa các file như pdf, csv, ..
 text_splitter: Dùng để chia file, kích thước các chunk
 webui_pages: Giao diện

