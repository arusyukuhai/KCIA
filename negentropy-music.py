import numpy as np
import zstd
import soundfile
import tqdm

def skew(data):
  n = len(data) # サンプルサイズ
  s = np.std(data) # 標準偏差
  mu = data.mean() # 平均
  return (n / ((n-1) * (n-2))) * (((data - mu) / s) ** 3).sum()

# 尖度の計算関数

def kurtosis(data):
  n = len(data) # サンプルサイズ
  s = np.std(data) # 標準偏差
  mu = data.mean() # 平均
  return ( ((n * (n+1)) / ((n-1) * (n-2) * (n-3))) * (((data - mu) ** 4) / (s ** 4)).sum() ) - ((3 * (n-1) ** 2) / ((n-2) * (n-3)))

def compress_length(array):
    return np.log(len(zstd.compress(bytes("".join([str(int(j)) for j in array]), "utf-8"), 6)) / len(array))

def sin_pi(wave_base):
    return np.sin(wave_base * np.pi)

def fm_synth(wave_base, _a):
    return sin_pi(wave_base * _a[0] + _a[1] * sin_pi(wave_base * _a[2] + _a[3] * sin_pi(wave_base * _a[4] + _a[5] * sin_pi(wave_base * _a[6]))) + _a[7] * sin_pi(wave_base * _a[8])) * 0.5 - 0.5

def melodic_symbolic_entropy(melody, edo):
    ts = []
    for j in range(1, 4):
        melody_ = melody[:len(melody)//j*j].reshape((-1, j)).T.flatten()
        ts.append(compress_length(floor_t(melody_)))
        ts.append(compress_length(floor_t(melody_[:-1] - melody_[1:])))
        ts.append(compress_length(floor_t(np.concatenate((np.abs(melody_[:-1] - melody_[1:]), np.sign(melody_[:-1] - melody_[1:]))))))
        ts.append(compress_length(floor_t(np.concatenate((melody_//np.ceil(np.sqrt(edo)), melody_%np.ceil(np.sqrt(edo)))))))
        ts.append(compress_length(floor_t(np.concatenate(((melody_[:-1] - melody_[1:])//np.ceil(np.sqrt(edo)), (melody_[:-1] - melody_[1:])%np.ceil(np.sqrt(edo)))))))
    return (np.square(skew(np.array(ts))) + np.square(kurtosis(np.array(ts)) - 3) / 4) - np.std(np.array(ts)) + np.mean(ts)

def edo_melody_to_wave(melody, edo, len=64):
    wave_base = []
    for tone in melody:
        wave_base.append(np.ones(len) * 2 ** (tone / edo - 1))
    wave_base = np.cumsum(wave_base)
    return wave_base

#def chordic_entropy(chords, edo):
#    

def floor_t(wave, q=32):
    return np.floor(wave * q)

def melodic_wavic_entropy(melody, edo):
    ts = []
    for j in range(1, 4):
        melody_ = melody[:len(melody)//j*j].reshape((-1, j)).T.flatten()
        wave_base = edo_melody_to_wave(melody_, edo)
        for jt1 in range(-edo//2, edo//2):
            wave = np.sin(wave_base * (2 ** jt1) * np.pi) * 0.5 - 0.5
            #ts.append(compress_length(floor_t(edo_melody_to_wave(melody_, edo))))
            for jt in range(1, 32):
                wave_ = wave[:len(wave)//jt*jt].reshape((-1, jt)).T.flatten()
                ts.append(compress_length(floor_t(np.nan_to_num(wave_))))
    return (np.square(skew(np.array(ts))) + np.square(kurtosis(np.array(ts)) - 3) / 4) - np.std(np.array(ts)) + np.mean(ts)

def melodic_entropy(melody, edo):
    return melodic_wavic_entropy(melody, edo) + melodic_symbolic_entropy(melody, edo)
#print(len([0,0,7,7,9,9,7,0,5,5,4,4,2,2,0,0,7,7,5,5,4,4,2,2,7,7,5,5,4,4,2,2,0,0,7,7,9,9,7,0,7,7,5,5,4,4,2,2,]))

a = []
scores = []
for j in tqdm.tqdm(range(32)):
    a.append(np.random.uniform(0, 12, 64))
    scores.append(melodic_entropy(np.floor(a[j]), 12))

print(np.min(scores), np.mean(scores))

for iter in range(1000):
    for d in tqdm.tqdm(range(32)):
        newa = np.nan_to_num(np.copy(a[d]))
        r1 = np.random.randint(0, len(a))
        r2 = np.random.randint(0, len(a))
        if(np.random.uniform(0, 1) < 0.2):
            newa = np.fft.ifft(np.fft.fft(newa + 0j) * np.fft.fft(a[r1] + 0j) / np.fft.fft(a[r2] + 0j)).real
        else:
            newa = newa + (a[r1] - a[r2]) * 0.5
        posr1 = np.random.randint(0, len(newa)-2)
        posr2 = np.random.randint(posr1, len(newa))
        newa[posr1:posr2] = a[d][posr1:posr2]
        newa = np.minimum(np.maximum(newa, 0), 12)
        score = melodic_entropy(np.nan_to_num(np.floor(newa)), 12)
        if(scores[d] >= score):
            a[d] = newa
            scores[d] = score
    print(np.min(scores), np.mean(scores))
    soundfile.write("a.wav", np.tanh(10 * np.sin(edo_melody_to_wave(a[np.argmin(scores)], 12, len=44100//6) / 64 * np.pi)) * 0.25, 44100)

"""#soundfile.write("a.wav", edo_melody_to_wave(np.random.randint(0, 7, 64) - 32, 7, 11025), 44100)
print(melodic_entropy(np.array([0,0,2,0,4,0,5,0,4,0,2,0,0,0,]), 12))
print(melodic_entropy(np.array([0,0,2,0,4,0,5,0,4,4,2,2,0,0,]), 12))
print(melodic_entropy(np.random.randint(0, 12, 48), 12))
print(melodic_entropy(np.random.randint(0, 12, 48), 12))
print(melodic_entropy(np.random.randint(0, 11, 11), 11))
print(melodic_entropy(np.random.randint(0, 22, 32), 11))
print(melodic_entropy(np.arange(12), 12))
print(melodic_entropy((np.arange(12)*7)%12, 12))
int(melodic_entropy((np.arange(12)*5)%12, 12))
print(melodic_entropy(np.sort(np.arange(7)*7), 12))
print(melodic_entropy(np.tile(np.sort(np.arange(7)*7), 4), 12))
print(melodic_entropy(np.array([0,0,7,7,9,9,7,0,5,5,4,4,2,2,0,0]), 12))
print(melodic_entropy(np.array([0,7,9,7,5,4,2,0]), 12))
print(melodic_entropy(np.array([0,0,7,7,9,9,7,0,5,5,4,4,2,2,0,0,7,7,5,5,4,4,2,2,7,7,5,5,4,4,2,2,0,0,7,7,9,9,7,0,7,7,5,5,4,4,2,2,]), 12))
print(melodic_entropy(np.array([0,0,7,7,9,9,7,0,5,5,4,4,2,2,0,0,7,7,5,5,4,4,2,2,7,7,5,5,4,4,2,2,0,0,7,7,9,9,7,0,7,7,5,5,4,4,2,2,]), 11))"""