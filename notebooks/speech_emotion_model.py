#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import librosa


# In[2]:


# [1] 데이터셋 구성 - TESS + RAVDESS + 감정 필터링

# 데이터셋 불러옴
TESS_path = r'C:\Users\dldpq\kimleepark\TESS Toronto emotional speech set data' # 본인 경로로 수정
RAVDESS_path = r'C:\Users\dldpq\kimleepark\Ravdess_by_emotion'

#사용할 감정 
selected_emotions = ['angry', 'happy', 'neutral', 'sad']


# In[3]:


# [2] 데이터셋 분류 - TESS
TESS_files, TESS_labels = [], []

for folder in os.listdir(TESS_path):
    folder_lower= folder.lower()
    for emotion in selected_emotions:
        if emotion in folder_lower:
            emotion_dir = os.path.join(TESS_path, folder)
            for wav in glob.glob(os.path.join(emotion_dir,"*.wav")):
                TESS_files.append(wav)
                TESS_labels.append(emotion)


# In[4]:


# [2] 데이터셋 분류 - RAVDESS
RAV_files, RAV_labels = [], []

for emotion in selected_emotions:
    emotion_dir = os.path.join(RAVDESS_path, emotion)
    for wav in glob.glob(os.path.join(emotion_dir,"*.wav")):
        RAV_files.append(wav)
        RAV_labels.append(emotion)


# In[5]:


from collections import Counter

# [3] 데이터셋 합치기 - TESS+RAVDESS
all_files = TESS_files + RAV_files
all_labels = TESS_labels + RAV_labels

print("전체 데이터 수 : ", len(all_files))

print("감정별 데이터 수")
counter = Counter(all_labels)
for emo, cnt in counter.items():
    print(f"{emo} : {cnt}")


# In[19]:


#file_path = r"C:\Users\dldpq\kimleepark\TESS Toronto emotional speech set data\OAF_angry\OAF_back_angry.wav"
#y, sr = librosa.load(file_path, sr=None) 
#print("TESS sample rate:", sr)

#file_path = r"C:\Users\dldpq\kimleepark\Ravdess_by_emotion\angry\Actor_01_03-01-05-01-01-01-01.wav"
#y, sr = librosa.load(file_path, sr=None)
#print("RAVDESS sample rate:", sr)


# In[6]:


# [4] 데이터 전처리
mfcc_list = []

for file_path in all_files:
    # 1. 오디오 불러오기 (16kHz, 모노)
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # 2. 길이 고정 (1.6초 = 25600 샘플)
    target_len = 16000 * 16 // 10   # = 25600
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # 3. MFCC 추출 (40차원, 25ms 창, 10ms 홉)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=40, n_fft=512,
        hop_length=160, win_length=400
    )  # shape: (40, T)

    # 4. 프레임 수 고정 (160프레임)
    if mfcc.shape[1] < 160:
        pad = np.zeros((40, 160 - mfcc.shape[1]))
        mfcc = np.hstack([mfcc, pad])
    else:
        mfcc = mfcc[:, :160]

    # 5. 정규화 (특징별 평균0, 표준편차1)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)

    mfcc_list.append(mfcc.astype(np.float32))

# 최종 데이터셋
processed_mfcc = np.stack(mfcc_list) 
print("MFCC shape:", processed_mfcc.shape)


# In[7]:


from sklearn.preprocessing import LabelEncoder

# [5] 라벨 인코딩 
label_encoder = LabelEncoder()
emotion_labels = label_encoder.fit_transform(all_labels).astype(np.int64)  
print("클래스 인덱스 :", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


# In[10]:


from sklearn.model_selection import train_test_split

# [6] 학습/검증 분할
X = np.transpose(processed_mfcc, (0, 2, 1)).astype(np.float32)
y = emotion_labels                                     

print("전체:", X.shape, y.shape)
print("전체 라벨 분포:", Counter(y))


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,         
    random_state=42,
    stratify=y             
)
print("\n1차 분할 완료")
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train 라벨 분포:", Counter(y_train))
print("Test  라벨 분포:", Counter(y_test))


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.15,        
    random_state=42,
    stratify=y_train
)
print("\n최종 세트")
print("Train(final):", X_tr.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Val 라벨 분포:", Counter(y_val))


# In[11]:


import tensorflow as tf
from tensorflow.keras import layers, models

# [7] Conv1D 모델 정의
num_classes = len(np.unique(y))

model = models.Sequential([
    layers.Input(shape=(160, 40)),                 

    layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPool1D(pool_size=2),                 

    layers.Conv1D(128, kernel_size=5, padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPool1D(pool_size=2),                 

    layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling1D(),               
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# In[12]:


# [8] 모델 학습
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix

# [9] 성능
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"최종 : loss={test_loss:.4f}, acc={test_acc:.4f}")

y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# In[16]:


model.save("speech_emotion_model.keras")

