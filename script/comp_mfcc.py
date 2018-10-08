import time
import numpy as np
import librosa
import Audio

start = time.time()

# (1) wavファイル読み込み
print("#1 [Wav files read]")

path1 = "wav/1_Piano_C3_Dur4_1.wav"
path2 = "wav/2_Piano_C3_Dur4_2.wav"
path3 = "wav/3_Piano_C3_Dur4_1_StartDelay.wav"
path4 = "wav/4_Piano_C3_Dur2.wav"
path5 = "wav/5_Piano_C3_Dur4_Reverb.wav"

fs1 = Audio.getSamplingFrequency(path1)
fs2 = Audio.getSamplingFrequency(path2)
fs3 = Audio.getSamplingFrequency(path3)
fs4 = Audio.getSamplingFrequency(path4)
fs5 = Audio.getSamplingFrequency(path5)

x1, fs1 = librosa.load(path1, fs1)
x2, fs2 = librosa.load(path2, fs2)
x3, fs3 = librosa.load(path3, fs3)
x4, fs4 = librosa.load(path4, fs4)
x5, fs5 = librosa.load(path5, fs5)

# (2) 特徴抽出
print("#2 [Feature extraction]")

mfcc1 = librosa.feature.mfcc(x1, fs1)
mfcc2 = librosa.feature.mfcc(x2, fs2)
mfcc3 = librosa.feature.mfcc(x3, fs3)
mfcc4 = librosa.feature.mfcc(x4, fs4)
mfcc5 = librosa.feature.mfcc(x5, fs5)

# (3) 類似度計算
print("#3 [Evaluation]")

ac1, wp1 = librosa.core.dtw(mfcc1, mfcc1)
ac2, wp2 = librosa.core.dtw(mfcc1, mfcc2)
ac3, wp3 = librosa.core.dtw(mfcc1, mfcc3)
ac4, wp4 = librosa.core.dtw(mfcc1, mfcc4)
ac5, wp5 = librosa.core.dtw(mfcc1, mfcc5)

eval1 = 1 - (ac1[-1][-1] / np.array(ac1).max())
eval2 = 1 - (ac2[-1][-1] / np.array(ac2).max())
eval3 = 1 - (ac3[-1][-1] / np.array(ac3).max())
eval4 = 1 - (ac4[-1][-1] / np.array(ac4).max())
eval5 = 1 - (ac5[-1][-1] / np.array(ac5).max())

print("    1 <--> 1:", round(eval1, 4))
print("    1 <--> 2:", round(eval2, 4))
print("    1 <--> 3:", round(eval3, 4))
print("    1 <--> 4:", round(eval4, 4))
print("    1 <--> 5:", round(eval5, 4))

end = time.time()
print("")
print("Eelapsed time:", str(round(end - start, 4)) + "[sec]")
