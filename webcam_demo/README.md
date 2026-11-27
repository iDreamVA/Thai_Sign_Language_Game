# ทดลองเว็บแคมภาษามือไทย

เปิดโฟลเดอร์นี้ใน VS Code แล้วเลือก Python interpreter เป็น `.venv/Scripts/python.exe` กด F5 เพื่อรันด้วย launch configuration ที่เตรียมไว้ (ใช้ส่วนขยาย Python และ Python Debugger ของ VS Code)
รันใน PowerShell จากโฟลเดอร์นี้

```powershell
.\.venv\Scripts\python.exe app.py
```

หรือใช้ `./run.ps1` โปรแกรมจะแสดงภาพเว็บแคม เส้นข้อต่อมือ ชื่อคำภาษาไทย และเปอร์เซ็นต์ความมั่นใจ เริ่มทำนายอัตโนมัติเมื่อเก็บภาพครบประมาณ 2 วินาที แล้วอัปเดตทุกประมาณ 0.3 วินาที กด Space เพื่อล้างช่วงท่าเดิม กด Esc หรือปุ่มปิดกล้องเพื่อออก

นั่งให้เห็นใบหน้า ไหล่ และมือทั้งสองข้าง ทำทีละคำให้จบในช่วงประมาณ 2 วินาที ถ้าใช้กล้องอีกตัวหรือทำท่าช้าลง

```powershell
.\.venv\Scripts\python.exe app.py --camera 1 --window-seconds 3
```

โฟลเดอร์ `model` มี TorchScript กับ metadata ที่คัดลอกจากโมเดล TSL-51 ของโปรเจกต์ รองรับ 52 คลาส รายชื่อคำดูใน `model/metadata.json` ถ้าเปลี่ยนโมเดลให้ใช้ไฟล์ทั้งคู่จากการส่งออกครั้งเดียวกัน พร้อมโค้ด preprocessing ที่ตรงกัน หรือระบุ `--model-dir` ไปยัง bundle อื่น

## เฟส 2 สแกนตามหมวดคำศัพท์

ถ้าเปิด PowerShell จากโฟลเดอร์โปรเจกต์หลัก ให้ `cd webcam_demo` ก่อน แล้วดูหมวดและคำที่โมเดลรองรับได้โดยไม่ต้องเปิดกล้อง

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py --list-categories
```

เปิดกล้องเฉพาะหมวดที่ต้องการ เช่น หมวดทักทายและมารยาทพื้นฐาน

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py --category greetings
```

| ค่าที่ใช้กับ `--category` | หมวด |
|---|---|
| `greetings` | ทักทายและมารยาทพื้นฐาน |
| `family` | แนะนำตัวเองและครอบครัว |
| `questions` | คำถามพื้นฐาน |
| `daily` | กิจกรรมในชีวิตประจำวัน |
| `feelings` | แสดงความรู้สึกต่อตัวเอง |
| `other` | คำอื่น ๆ ที่มีในโมเดล |

โปรแกรมพิมพ์รายชื่อคำของหมวดตอนเริ่ม และพิมพ์ผลพร้อมค่าความมั่นใจเมื่อทายได้คำใหม่ในหมวด ถ้าคำที่โมเดลทายอยู่นอกหมวด หน้าต่างกล้องจะแสดง `ไม่พบคำในหมวดนี้` คำที่มีหลายรูปท่า เช่น `ฉัน` ใช้ได้ทั้งสองรูป ดู `sign_id` ของแต่ละรูปด้วย `--list-words --category family`

## ตรวจโมเดลทีละคำ

เปิด Terminal ใหม่ใน VS Code แล้วรันคำสั่งต่อไปนี้ หากเริ่มจากโฟลเดอร์โปรเจกต์หลัก ให้พิมพ์ `cd webcam_demo` ก่อน

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py --target "ขอบคุณ" --trials 10
```

กล้องจะแสดงคำเป้าหมาย กด Space หรือปุ่มเริ่มรอบ จากนั้นเตรียมตัว 1.5 วินาทีและทำท่าหนึ่งครั้งในช่วงเก็บภาพ 2 วินาที โปรแกรมจะทำนายหนึ่งครั้ง บอกชื่อท่าที่เลือก เปอร์เซ็นต์ความมั่นใจ คะแนนของคำเป้าหมาย และผลตรงหรือไม่ตรง กด Space เพื่อทำรอบถัดไปจนครบ 10 รอบ ปิดหน้าต่างก่อนเปลี่ยนคำ

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py --target "สวัสดี" --trials 10
.\.venv\Scripts\python.exe -X utf8 app.py --target "สบายดี" --trials 10
.\.venv\Scripts\python.exe -X utf8 app.py --list-words
```

ให้รันทีละคำสั่ง ถ้าคำไทยหนึ่งคำมีหลายรูปท่า เช่น ฉัน ให้ระบุ sign_id เต็ม

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py --target "ฉัน_var_1" --trials 10
```

โมเดลจำแนกครบ 52 คลาสแล้วเทียบผลกับ sign_id ของคำเป้าหมาย ผลแต่ละรอบถูกบันทึกใน `sessions/*_word_test.csv` รวมคะแนนคำเป้าหมาย จำนวนเฟรม และผลตรงคำ ค่าความมั่นใจสูงยังเกิดขึ้นเมื่อทายผิดได้ คำเป้าหมายเป็นป้ายกำกับที่ผู้ทดสอบระบุ จึงควรให้ผู้ใช้ภาษามือช่วยตรวจว่าท่าที่ทำตรงกับรูปท่าในชุดข้อมูลด้วย

สัดส่วนตรงคำเป้าหมายจากการทำคำนั้นซ้ำแสดงการตรวจพบคำในเงื่อนไขที่ทดลอง การประเมิน precision/F1 ต้องมีทั้งคำนี้ คำอื่น และช่วงพัก เพื่อวัดการทายคำนี้ผิดโอกาสด้วย ทดลองหลายผู้เซ็น แสงและระยะกล้องต่างกัน และเก็บแต่ละคำจำนวนใกล้เคียงกันเพื่อประเมินการใช้งานจริง

ตรวจผลเดิมครบทุกคำ พร้อมดูคำที่สับสนบ่อย

```powershell
.\.venv\Scripts\python.exe -X utf8 review_model.py
```

รายงานจะอยู่ใน `reports/model_word_report.csv` คอลัมน์ recall คือสัดส่วนคลิปของคำนั้นที่ทายถูก ส่วน precision คือสัดส่วนที่ถูกเมื่อโมเดลทายเป็นคำนั้น F1 สรุปสองค่านี้ ผลเดิมมาจากชุด user_sign ของ TSL-51 ซึ่งมีผู้เซ็นหนึ่งคนและส่วนใหญ่ประมาณ 10 คลิปต่อคำ

เปอร์เซ็นต์บนจอคือ softmax confidence ของคำที่โมเดลเลือก คะแนนความถูกต้องของการทำท่าต้องมีข้อมูลอ้างอิงหรือเกณฑ์คุณภาพท่าเพิ่มเติม ผลทดสอบโมเดลเดิมกับผู้เซ็นในชุดทดสอบมี accuracy ประมาณ 53.75% ผลผ่านเว็บแคมขึ้นกับแสง มุมกล้อง ผู้เซ็น และการตัดช่วงท่า โมเดลนี้เรียนจากคลิปทีละคำ จึงให้ทำทีละท่าสำหรับการทดลอง

ภาพพรีวิวกลับซ้ายขวาเพื่อดูเหมือนกระจก การสกัด landmark และทำนายใช้ภาพต้นฉบับ โปรแกรมประมวลผลภาพในเครื่อง ภาพอยู่ในหน่วยความจำระหว่างเปิดโปรแกรม

## ติดตั้งใหม่

ใช้ Python 3.11 สำหรับ MediaPipe Holistic รุ่นที่ระบุใน requirements

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py --self-check
```

หากใช้ `uv` สามารถติดตั้ง Python และแพ็กเกจแยกในโฟลเดอร์นี้ได้

```powershell
uv python install 3.11 --install-dir .python
uv venv --python .python\cpython-3.11-windows-x86_64-none\python.exe .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## จุด landmark ที่ส่งเข้าโมเดล

ใช้โค้ด `tsl_preprocess.py` และ `tsl_inference.py` ของ bundle เดิม เรียงจุด 54 จุดตาม metadata เป็นมือซ้าย 21 จุด มือขวา 21 จุด pose 11–16 และใบหน้า 105, 70, 300, 334, 61, 291 ตามลำดับในคำอธิบายชุดข้อมูล ถ้าจับจุดไม่ได้ใช้ NaN แล้วแปลงเป็น 48 เฟรม × 216 features แบบเดียวกับตอนเทรน

อ้างอิง [TSL-51 landmark extraction](https://huggingface.co/datasets/Namonpas/thai-sign-language-tsl51/blob/main/README.md#landmark-extraction-mediapipe) และ [MediaPipe Holistic](https://chuoling.github.io/mediapipe/solutions/holistic.html)

## Windows: Tk DLL blocked by Application Control

The app automatically uses an OpenCV window if Tk cannot load. Select it explicitly with:

```powershell
.\.venv\Scripts\python.exe -X utf8 app.py --ui opencv
```

The OpenCV window supports live predictions and `--target` / `--trials` CSV recording. Focus the window and press Space to reset or start a trial; press Esc or close the window to quit. `--ui tk` requires working Tk DLLs. No Windows security policy changes are needed for the OpenCV backend.

`--list-words`, `--self-check`, and `check_pipeline.py` do not require Tk. A `.pt` model is a binary TorchScript file; use `--self-check` to validate it instead of opening it as text.
