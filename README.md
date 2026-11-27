# Thai Sign Language Game

โปรเจกต์ทดลองรู้จำคำภาษามือไทยผ่านเว็บแคม ใช้ MediaPipe จับจุดบนร่างกายและโมเดล LSTM จำแนก 52 คลาสจากชุดข้อมูล TSL‑51 (51 รูปท่าและคลาส `null_act`) โมเดลที่เทรนแล้วอยู่ในโปรเจกต์ เปิดกล้องและทดสอบได้ด้วย CPU โดยใช้ชุดข้อมูล TSL‑51 เพิ่มเฉพาะตอนเทรนใหม่

## 1. ดาวน์โหลดโปรเจกต์

คู่มือนี้ใช้ **Windows 10/11 แบบ x64 (Intel/AMD)** และ **PowerShell** เตรียมอินเทอร์เน็ตสำหรับติดตั้งครั้งแรกและเว็บแคมสำหรับโหมดกล้อง

วิธีเร็วที่สุดคือเปิดหน้า [Thai Sign Language Game บน GitHub](https://github.com/iDreamVA/Thai_Sign_Language_Game) กด **Code → Download ZIP** แตกไฟล์ แล้วเปิด PowerShell ในโฟลเดอร์ที่มี `webcam_demo` และ `tsl51_lstm_bundle` อยู่ด้วยกัน

ถ้าจะใช้ `git clone` ให้ติดตั้ง [Git for Windows](https://git-scm.com/install/windows) ก่อน หากมี Git อยู่แล้วให้ข้ามคำสั่งติดตั้ง

```powershell
winget install --id Git.Git --exact --source winget
```

เปิด PowerShell ใหม่ในโฟลเดอร์ที่จะเก็บโปรเจกต์ แล้วรัน

```powershell
git clone https://github.com/iDreamVA/Thai_Sign_Language_Game.git
cd Thai_Sign_Language_Game
```

หาก GitHub ขอสิทธิ์ ให้เข้าสู่ระบบด้วยบัญชีที่เข้าถึง repo ได้ หรือขอ ZIP ทั้งโปรเจกต์จากเจ้าของ ถ้าต้องการโค้ดที่เจ้าของกำลังแก้อยู่ ให้ขอไฟล์เวอร์ชันนั้นด้วย เพราะ GitHub มีเฉพาะเวอร์ชันที่ push ขึ้นไปแล้ว ให้สร้าง `.venv` บนเครื่องตัวเองตามข้อถัดไป

## 2. ติดตั้ง Python และไลบรารี

ใช้ **Python 3.11 x64** (เครื่องที่ทดสอบใช้ **3.11.15**) ให้ `uv` จัดการ Python และไลบรารี ถ้ายังไม่มี `uv` ให้รันคำสั่งนี้แล้วเปิด PowerShell ใหม่ [คู่มือติดตั้ง uv](https://docs.astral.sh/uv/getting-started/installation/#winget)

```powershell
winget install --id astral-sh.uv --exact --source winget
```

จากโฟลเดอร์หลักของโปรเจกต์ รันทีละบรรทัด

```powershell
uv --version
uv python install 3.11
uv venv --python 3.11 --managed-python .\webcam_demo\.venv
uv pip install --python .\webcam_demo\.venv\Scripts\python.exe -r .\webcam_demo\requirements.txt matplotlib scikit-learn
uv pip check --python .\webcam_demo\.venv\Scripts\python.exe
.\webcam_demo\.venv\Scripts\python.exe -c "import sys, platform; print(sys.version.split()[0], platform.machine())"
```

บรรทัดสุดท้ายควรแสดง `3.11.x AMD64` โดย `AMD64` คือชื่อสถาปัตยกรรม x64 ทั้งบน Intel และ AMD ไฟล์ [`requirements.txt`](webcam_demo/requirements.txt) ระบุเวอร์ชัน MediaPipe `0.10.21`, NumPy `1.26.4`, OpenCV Contrib `4.11.0.86`, Pillow `11.3.0` และ PyTorch `2.7.1` ส่วน `matplotlib` และ `scikit-learn` ใช้สร้างกราฟผลโมเดล คำสั่งด้านล่างเรียก Python ใน `.venv` โดยตรง จึงใช้ได้ทันทีเมื่อเปิด Terminal ใหม่ที่โฟลเดอร์หลัก

<details>
<summary>ติดตั้งด้วย Python และ pip โดยตรง</summary>

ติดตั้ง Python 3.11 x64 พร้อม `pip`, `venv` และ Python Launcher แล้วเปิด PowerShell ใหม่ หากใช้ตัวติดตั้งแบบ `.exe` ให้เลือก **Windows installer (64-bit)** จากหน้า [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)

```powershell
py -3.11 --version
py -3.11 -m venv .\webcam_demo\.venv
.\webcam_demo\.venv\Scripts\python.exe -m pip install --upgrade pip
.\webcam_demo\.venv\Scripts\python.exe -m pip install -r .\webcam_demo\requirements.txt matplotlib scikit-learn
.\webcam_demo\.venv\Scripts\python.exe -m pip check
```

</details>

## 3. ตรวจการติดตั้งและเปิดกล้อง

สองคำสั่งแรกตรวจโมเดล การจับจุด และการบันทึกผลด้วยข้อมูลจำลอง คำสั่งแรกควรขึ้น `OK: MediaPipe, TorchScript, 52 classes, 48 frames, 216 features` ส่วนคำสั่งที่สองควรขึ้น `PASS` คำสั่งที่สามจะเปิดกล้องจริง

```powershell
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --self-check
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\check_pipeline.py
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py
```

ให้กล้องเห็นหน้า ไหล่ และมือทั้งสองข้าง ทำทีละคำประมาณ 2 วินาที กด `Space` เพื่อล้างช่วงท่าแล้วเริ่มจับใหม่ กด `Esc` เพื่อออก ปิดหน้าต่างกล้องก่อนลองคำสั่งถัดไป

เปิดแบบสั้นได้ด้วย `.\webcam_demo\run.ps1` ถ้าใช้ VS Code ให้เปิดโฟลเดอร์ `webcam_demo` โดยตรง ติดตั้งส่วนขยาย **Python** และ **Python Debugger** ของ Microsoft เลือก `.venv\Scripts\python.exe` เป็น interpreter แล้วกด `F5` ที่รายการ **Thai Sign Language Webcam** [คู่มือ Python ใน VS Code](https://code.visualstudio.com/docs/python/python-quick-start)

## 4. คำสั่งที่ใช้ใน Terminal

รันทีละคำสั่งจากโฟลเดอร์หลัก แต่ละบรรทัดใช้ได้ทันทีหลังติดตั้งเสร็จ

```powershell
# ดูคำสั่งและตัวเลือกทั้งหมด
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --help

# ดูหมวด คำศัพท์ทั้งหมด และ sign_id เฉพาะหมวดครอบครัว
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --list-categories
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --list-words
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --list-words --category family

# เปิดกล้องและแสดงผลเฉพาะหมวดทักทาย
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --category greetings

# ทดสอบคำเป้าหมาย 10 รอบ แล้วเก็บผลเป็น CSV
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --target "ขอบคุณ"
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --target "ฉัน_var_1" --results-dir .\webcam_demo\sessions

# ใช้กล้องตัวที่สอง จับท่า 3 วินาที และเปิดหน้าต่าง OpenCV
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --camera 1 --window-seconds 3 --ui opencv

# ตรวจโมเดลจาก bundle ที่ระบุ ก่อนนำมาใช้กับกล้อง
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\app.py --model-dir .\tsl51_lstm_bundle --self-check

# สร้างรายงานรายคำจากผลทดสอบเดิม
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\webcam_demo\review_model.py

# สร้างกราฟผลการเทรนและประเมินโมเดล
.\webcam_demo\.venv\Scripts\python.exe -X utf8 .\plot_tsl_metrics.py .\tsl51_lstm_bundle
```

- หมวดที่ใช้กับ `--category` มี `greetings`, `family`, `questions`, `daily`, `feelings` และ `other`
- ในโหมด `--target` กด `Space` เพื่อเริ่มแต่ละรอบ โปรแกรมให้เตรียมตัว 1.5 วินาที แล้วเก็บภาพ 2 วินาทีตามค่าเริ่มต้น ใช้ `sign_id` เต็มเมื่อคำหนึ่งมีหลายรูปท่า ผลอยู่ใน `webcam_demo/sessions/*_word_test.csv`
- ค่าเริ่มต้นของกล้องคือ `--camera 0` และ `--ui auto` ซึ่งจะใช้ OpenCV หากเปิด Tk ไม่ได้ `--window-seconds` ตั้งได้ตั้งแต่ `0.5` ถึง `10`
- `review_model.py` เขียนรายงานที่ `webcam_demo/reports/model_word_report.csv` ใช้ `--output` เมื่อต้องการชื่อไฟล์อื่น หรือ `--bundle` เมื่ออ่านผลจาก bundle อื่น
- `plot_tsl_metrics.py` บันทึก PNG และ PDF ใน `tsl51_lstm_bundle/plots/` และอาจเขียนทับกราฟเดิม เพิ่ม `--show` หากต้องการเปิดกราฟบนจอด้วย
- `--model-dir` ต้องชี้ไปยังโฟลเดอร์ที่มี `model.torchscript.pt` และ `metadata.json` จากการ export ครั้งเดียวกัน โมเดลต้องใช้การเตรียม landmark แบบเดียวกับตัวเว็บแคม

รายละเอียดโหมดกล้องและการอ่านผลอยู่ใน [คู่มือ webcam_demo](webcam_demo/README.md)

## 5. เทรนโมเดลใหม่ใน Google Colab

1. เลือก [`train_tsl51_lstm_colab.ipynb`](train_tsl51_lstm_colab.ipynb) สำหรับเทรน LSTM หรือ [`train_tsl51_plain_lstm_colab.ipynb`](train_tsl51_plain_lstm_colab.ipynb) สำหรับเปรียบเทียบฟีเจอร์สามแบบใน LSTM ทิศทางเดียว
2. อัปโหลดไฟล์ที่เลือกใน [Google Colab](https://colab.research.google.com/) เลือก **Runtime → Change runtime type → GPU** แล้วกด **Run all**
3. ค่า `DOWNLOAD_TSL51 = "training"` จะดาวน์โหลดเฉพาะข้อมูลที่ใช้เทรนและวัดผลให้อัตโนมัติ ถ้ามีข้อมูลอยู่แล้ว ให้ตั้งเป็น `"local"` และชี้ `TSL51_DIR` ไปยังโฟลเดอร์ที่มี `metadata/` และ `landmarks/` [ไฟล์และเงื่อนไขชุดข้อมูล TSL‑51](https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51/blob/main/README.md)
4. เมื่อรันเสร็จ ดูพาธ ZIP จากผลของเซลล์ export แล้วดาวน์โหลดจากแถบ **Files** ของ Colab หรือเปลี่ยน `DOWNLOAD_BUNDLE = True` แล้วรันเซลล์ดาวน์โหลด หากต้องการบันทึกลง Google Drive ให้ตั้ง `SAVE_TO_DRIVE = True` ในเซลล์ตั้งค่าก่อนรันและอนุญาตให้ Colab เข้าถึง Drive

## แก้ปัญหาที่พบบ่อย

| อาการ | วิธีแก้ |
| --- | --- |
| PowerShell หา `git` หรือ `uv` ไม่เจอหลังติดตั้ง | ปิดแล้วเปิด PowerShell หรือ VS Code ใหม่ ถ้าไม่มี `winget` ให้ใช้ตัวติดตั้งจากลิงก์ในข้อ 1–2 |
| หา `.venv` หรือไฟล์ `.py` ไม่เจอ | ใช้ `Get-Location` เช็กว่าอยู่ในโฟลเดอร์หลัก แล้วรันข้อ 2 ให้เสร็จ |
| `No module named ...` หรือ `No matching distribution` | เช็กว่าใช้ `.venv\Scripts\python.exe` ใน `webcam_demo` และเป็น Python 3.11 x64 แล้วรันคำสั่งติดตั้งไลบรารีในข้อ 2 อีกครั้ง |
| `DLL load failed` ตอน import `torch`, `cv2` หรือ `mediapipe` | เช็ก Python x64 และติดตั้งหรือ Repair [Microsoft Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist#latest-supported-redistributable-version) หาก DLL ที่แจ้งคือ Tk ให้ลอง `--ui opencv` |
| เปิดเว็บแคมไม่ได้ | เปิดสิทธิ์กล้องและ **Let desktop apps access your camera** ใน Windows Settings ปิดโปรแกรมอื่นที่ใช้กล้อง แล้วลอง `--camera 1` ถ้ามีมากกว่าหนึ่งตัว |
| PowerShell บล็อก `run.ps1` | ใช้คำสั่งที่เรียก `app.py` ผ่าน `.venv\Scripts\python.exe` โดยตรงจากข้อ 3 |
| หา `model.torchscript.pt` หรือ `metadata.json` ไม่เจอ | เช็กว่า ZIP แตกครบและไฟล์ทั้งคู่อยู่ใน `webcam_demo/model/` หรือใช้ `--model-dir` ไปยัง bundle ที่ครบ |
