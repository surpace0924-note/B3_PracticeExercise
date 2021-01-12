R_motor ロボットの指令に関するプログラム
(2021-01-12現在修正中)

・RaspberryMotor.py:
  HTTP経由でロボットに指令を出すプログラム
    
    http://rpi_addr:port/?action=x      
    where x is one of [forward, backward, left, right, stop]
  
・encoder.py:
   ロボットの回転角を取得するエンコーダのプログラム

・brick_pi_sample.py:
　ロボット（brickpi）を動かすサンプルプログラム

・Computer_motor.py:
コンピュータの処理を行うプログラム例
画像を取得して走行指令を送る（現在はランダムに動く設定）

注：RとSの接続部分に関しては現在修正中なので後日提供予定
    
