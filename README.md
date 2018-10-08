# Wave-Comparator
2つの波形を比較するPythonスクリプト。

### 必要なもの
* Python 3.7.0以上の実行環境
  * numpy
  * librosa

### 概要
2つの波形から以下の特徴量を抽出し、DTW（動的時間短縮）によって差分を計算、類似度を算出します。互いに時間長が異なる波形の比較も可能です。
* 振幅スペクトル
* 振幅スペクトル（セントロイド）
* MFCC（メル周波数ケプストラム係数）

### 実行例
```
#1 [Wav files read]
#2 [Feature extraction]
#3 [Evaluation]
1 <--> 1: 1.0
1 <--> 2: 0.9862
1 <--> 3: 0.9492
1 <--> 4: 0.9054
1 <--> 5: 0.7837
Eelapsed time: 0.5024[sec]
```
