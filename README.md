# MEDISCAN

## 📋 프로젝트 소개

MEDISCAN은 의약품 이미지를 분석하여 식별 정보를 제공하는 웹 애플리케이션입니다. 사용자가 의약품 이미지를 업로드하면 다음과 같은 정보를 분석합니다:

- ✅ 의약품 감지 및 식별
- 🎨 색상 분석
- 📐 모양 분석
- 📝 텍스트 인식 (OCR)
- 🔍 재질 분석

## 🏗️ 프로젝트 구조

```
MEDISCAN/
├── frontend/               # Next.js 프론트엔드 애플리케이션
│   ├── src/
│   │   ├── app/           # Next.js 앱 라우터
│   │   ├── components/    # React 컴포넌트
│   │   └── lib/          # 유틸리티 함수 및 라이브러리
│   ├── public/           # 정적 파일
│   └── package.json      # 프론트엔드 의존성
│
├── backend/              # FastAPI 백엔드 서버
│   ├── main.py          # 백엔드 메인 코드
│   ├── yolov8n.pt       # YOLO 머신러닝 모델
│   └── requirements.txt  # 백엔드 의존성
│
└── .venv/               # Python 가상 환경
```

## 🚀 시작하기

### 필수 요구사항

- Node.js 18.0.0 이상
- Python 3.8 이상
- pip (Python 패키지 관리자)
- npm (Node.js 패키지 관리자)

### 설치 및 실행

1. **프론트엔드 설정**

```bash
# 프론트엔드 디렉토리로 이동
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

프론트엔드는 [http://localhost:3000](http://localhost:3000)에서 확인할 수 있습니다.

2. **백엔드 설정**

```bash
# 백엔드 디렉토리로 이동
cd backend

# 가상 환경 생성 및 활성화
# Windows
python -m venv .venv
.\.venv\Scripts\activate
# MacOS/Linux
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

백엔드 API는 [http://localhost:8000](http://localhost:8000)에서 제공됩니다.

## 💻 기술 스택

### 프론트엔드
- [Next.js 15.3.0](https://nextjs.org) - React 프레임워크
- [React 19](https://react.dev) - UI 라이브러리
- [Tailwind CSS](https://tailwindcss.com) - CSS 프레임워크
- [TypeScript](https://www.typescriptlang.org) - 정적 타입 지원

### 백엔드
- [FastAPI](https://fastapi.tiangolo.com) - 백엔드 API 프레임워크
- [Ultralytics YOLO v8](https://github.com/ultralytics/ultralytics) - 객체 탐지 모델
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - 광학 문자 인식
- [OpenCV](https://opencv.org) - 이미지 처리

## 🛠️ API 문서

### 엔드포인트

- `GET /` - API 상태 확인
- `POST /analyze` - 이미지에서 의약품 감지
- `POST /analyze-pill` - 의약품 세부 분석 (색상, 모양, 텍스트 등)

### API 사용 예시

```python
import requests

# 이미지 분석 요청
response = requests.post(
    "http://localhost:8000/analyze",
    files={"file": open("pill_image.jpg", "rb")}
)

# 응답 확인
print(response.json())
```

## 🧪 테스트

### 프론트엔드 테스트
```bash
cd frontend
npm test
```

### 백엔드 테스트
```bash
cd backend
pytest
```

## 🤝 기여하기

1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. Pull Request를 생성합니다.

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

