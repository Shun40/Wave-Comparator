import wave

# ===
# wavファイルからサンプリング周波数を抽出する
# ===
# path : wavのファイルパス
def getSamplingFrequency(path):
    wr = wave.open(path, "r")
    fs = wr.getframerate()
    wr.close()
    return fs
