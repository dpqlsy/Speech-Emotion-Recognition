Speech Emotion Recognition (Keras + Librosa)

음성 신호로부터 감정을 분류하는 딥러닝 모델입니다.  
MFCC 특징을 추출하고 CNN 모델을 학습하여 감정을 예측합니다.

---

프로젝트 개요
- Framework: TensorFlow / Keras  
- Features: MFCC (librosa 기반 40차원)  
- Model: Conv1D 기반 CNN  
- Emotions: Happy / Sad / Angry / Neutral  
- Dataset: RAVDESS, TESS 등 (사용자 확장 가능)

---

코드 구조
| 폴더/파일 | 설명 |
| `notebooks/speech_emotion_model.ipynb` | 모델 학습 코드 |
| `notebooks/predict_emotion_audio.ipynb` | 학습된 모델로 예측 |
| `models/speech_emotion_model.keras` | 훈련된 모델 |
| `requirements.txt` | 의존성 패키지 목록 |

---

설치 및 실행
```bash
git clone https://github.com/dpqlsy/Speech-Emotion-Recognition.git
cd SpeechEmotionRecognition
pip install -r requirements.txt
