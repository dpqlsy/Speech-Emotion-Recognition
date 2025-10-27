#!/usr/bin/env python
# coding: utf-8

# # 🎤 실시간 마이크 입력 음성 감정 분류 (speech_emotion_model 기반)
# 
# - 모델 파일: **speech_emotion_model.keras**
# - 샘플레이트: **16000 Hz**
# - 특징: **MFCC 40** (`n_fft=512`, `hop_length=160`, `win_length=400`)
# - 세그먼트 길이: **2.0초**
# - 레이블: **['angry', 'happy', 'neutral', 'sad']**
# - 입력형태: **Conv1D (2D 아님)**
# 
# > 이 노트북은 `speech_emotion_model.ipynb`로 학습한 모델(**speech_emotion_model.keras**)을 사용하여 **실시간 마이크 입력**을 분류합니다.

# In[ ]:


# !pip install sounddevice librosa tensorflow
# Windows에서 WASAPI 장치 접근 문제가 있으면 관리자 권한 또는 장치 설정을 확인하세요.


# In[ ]:


MODEL_PATH = "speech_emotion_model.keras"
LABELS = ['angry', 'happy', 'neutral', 'sad']

SR = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
SEGMENT_SEC = 2.0
EXPECTS_2D = False

# 슬라이딩 윈도우 간격(초). 0.5면 2초 창을 1초 겹치게 이동
SLIDE_SEC = 0.5
TOPK = 3  # 상위 K개의 확률 출력
SMOOTHING = 0.6  # 지수이동평균(EMA) 알파


# In[ ]:


import numpy as np
import sounddevice as sd
import librosa, queue, threading, time, sys, json
import tensorflow as tf
from collections import deque

def softmax(x, axis=-1):
    x = np.asarray(x)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)

def extract_feature(y, sr):
    # MFCC
    feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                                hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    # 샘플 내 표준화
    feat = (feat - feat.mean(axis=None, keepdims=True)) / (feat.std(axis=None, keepdims=True) + 1e-6)
    return feat

def prepare_input(feat):
    X = np.array([feat], dtype=np.float32)
    # Conv1D 입력: (batch, time, feat) 형태로 전치
    if X.ndim == 3:
        X = np.transpose(X, (0, 2, 1))
    return X


# In[ ]:


model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model:", MODEL_PATH)


# In[ ]:


def predict_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    seg_len = int(SEGMENT_SEC * SR)
    if len(y) < seg_len:
        pad = np.zeros(seg_len, dtype=y.dtype); pad[:len(y)] = y; y = pad
    else:
        y = y[:seg_len]
    y = y / (np.abs(y).max() + 1e-9)
    feat = extract_feature(y, SR)
    X = prepare_input(feat)
    logits = model.predict(X, verbose=0)
    prob = softmax(logits)[0]
    top_idx = int(np.argmax(prob))
    result = { "pred": LABELS[top_idx],
               "probs": {lbl: float(p) for lbl, p in zip(LABELS, prob)} }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

# 사용 예시:
# predict_wav("sample.wav")


# In[ ]:


class RealtimeEmotion:
    def __init__(self, samplerate=SR, segment_sec=SEGMENT_SEC, slide_sec=SLIDE_SEC):
        self.sr = samplerate
        self.segment_len = int(segment_sec * samplerate)
        self.slide_len = int(slide_sec * samplerate)
        self.buffer = deque(maxlen=self.segment_len)  # 고정 창
        self.q = queue.Queue()
        self.stream = None
        self.ema = np.zeros(len(LABELS), dtype=np.float32)

    def _callback(self, indata, frames, time_info, status):
        if status:
            # 상태 경고(언더런/오버런) 무시 가능
            pass
        # mono
        x = indata.copy().astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        self.q.put(x)

    def start(self, device=None):
        self.stream = sd.InputStream(callback=self._callback, channels=1,
                                     samplerate=self.sr, device=device, blocksize=self.slide_len)
        self.stream.start()
        print("🎙️ Listening... (Ctrl+C to stop)")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def run(self):
        try:
            while True:
                x = self.q.get()
                for s in x:
                    self.buffer.append(float(s))
                # 창이 충분히 찼을 때만 추론
                if len(self.buffer) == self.segment_len:
                    y = np.array(self.buffer, dtype=np.float32)
                    # 정규화
                    if np.abs(y).max() > 0:
                        y = y / (np.abs(y).max() + 1e-9)
                    feat = extract_feature(y, self.sr)
                    X = prepare_input(feat)
                    logits = model.predict(X, verbose=0)
                    prob = softmax(logits)[0]
                    # EMA smoothing
                    self.ema = SMOOTHING * self.ema + (1.0 - SMOOTHING) * prob
                    topk_idx = np.argsort(self.ema)[-TOPK:][::-1]
                    msg = " | ".join([f"{LABELS[i]}: {self.ema[i]:.3f}" for i in topk_idx])
                    sys.stdout.write(f"\r{msg}      ")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
        finally:
            self.stop()

# 장치 목록 확인 (필요시)
# print(sd.query_devices())


# In[ ]:


rt = RealtimeEmotion()
rt.start(device=None)  # 특정 장치를 쓰려면 인덱스/이름을 지정
rt.run()

