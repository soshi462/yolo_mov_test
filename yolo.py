import cv2
from ultralytics import YOLO
import torch
import threading

# 使用するデバイスを選択 (GPUが利用可能ならCUDA、なければCPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# YOLOモデルをロード
model = YOLO("yolo11n.pt")
model.to(device)  # モデルを指定したデバイスに移動

# カメラを開く (0はデフォルトのカメラを指定)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")  # カメラが開けなかった場合のエラーメッセージ
    exit()

# カメラの解像度を設定 (幅640px、高さ480px)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 共有データ (フレーム) と制御用フラグ
frame = None  # 最新のカメラ画像を格納
lock = threading.Lock()  # スレッド間のデータ保護用ロック
stop_flag = False  # ループを停止するためのフラグ

# カメラから画像を取得するスレッド処理
def camera_reader():
    global frame, stop_flag
    while not stop_flag:
        ret, temp_frame = camera.read()  # カメラからフレームを取得
        if ret:
            with lock:  # ロックを使用してデータの競合を防ぐ
                frame = temp_frame  # 取得したフレームを共有変数に保存

# カメラ読み取りをバックグラウンドで実行するスレッドを開始
thread = threading.Thread(target=camera_reader)
thread.start()

print("Press 'q' to quit.")  # 終了するための案内メッセージ

try:
    while True:
        # 最新のフレームを取得
        with lock:
            if frame is None:  # まだフレームが取得されていない場合はスキップ
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB に変換 (YOLO用)

        # YOLOで物体検出を実行
        results = model(rgb_frame, imgsz=320, conf=0.5)

        # 検出結果を描画 (YOLOのplot関数を使用)
        annotated_frame = results[0].plot()

        # 検出結果を画面に表示
        cv2.imshow("YOLO Detection", annotated_frame)

        # 'q'キーが押されたらループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag = True
            break

finally:
    # プログラム終了時の後処理
    stop_flag = True  # スレッド停止フラグをセット
    thread.join()  # スレッドの終了を待機
    camera.release()  # カメラを解放
    cv2.destroyAllWindows()  # ウィンドウを閉じる
