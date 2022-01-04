# mediapipe-hand-X-linebot
這是我把mediapipe手勢辨識跟linebot結合

1.當大拇指跟食指伸出時，會拍一張照但不辨識人臉，linebot傳照片給line user

2.當食指跟中指伸出，會拍一張照且辨識人臉，linebot傳照片給line user

3.至於其他手勢，則是linebot傳手勢的文字訊息給line user

4.操作順序:

  (1) 一開始要先執行 "enroll_user_id.py" 去登錄 line user id 
      (user 傳 "enroll"給 line bot)
  (2) 再用 "Project_for_拍照_linebot傳網站的照片到line_linebot傳送辨識的手勢以文字訊息到line.py " 去跑手勢辨識跟傳圖片
