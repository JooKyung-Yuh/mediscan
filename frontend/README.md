# MEDISCAN

MEDISCAN은 의약품 이미지를 분석하여 식별 정보를 제공하는 웹 애플리케이션입니다.

## 📋 프로젝트 개요

이 애플리케이션은 사용자가 의약품 이미지를 업로드하면 다음과 같은 정보를 분석합니다:
- 의약품 감지 및 식별
- 색상 분석
- 모양 분석
- 텍스트 인식 (OCR)
- 재질 분석

## 🏗️ 프로젝트 구조

```
MEDISCAN/
├── frontend/               # Next.js 프론트엔드 애플리케이션
│   ├── src/
│   │   ├── app/            # Next.js 앱 라우터
│   │   ├── components/     # React 컴포넌트
│   │   └── lib/            # 유틸리티 함수 및 라이브러리
│   ├── public/             # 정적 파일
│   └── package.json        # 프론트엔드 의존성
│
├── backend/                # FastAPI 백엔드 서버
│   ├── main.py             # 백엔드 메인 코드
│   ├── yolov8n.pt          # YOLO 머신러닝 모델
│   └── requirements.txt    # 백엔드 의존성
│
└── .venv/                  # Python 가상 환경
```

## 🚀 시작하기

### 프론트엔드 실행

```bash
# 프론트엔드 디렉토리로 이동
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

프론트엔드는 [http://localhost:3000](http://localhost:3000)에서 확인할 수 있습니다.

### 백엔드 실행

```bash
# 백엔드 디렉토리로 이동
cd backend

# 가상 환경 활성화 (선택 사항)
# Windows
.\.venv\Scripts\activate
# MacOS/Linux
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

백엔드 API는 [http://localhost:8000](http://localhost:8000)에서 제공됩니다.

## 💻 주요 기술 스택

### 프론트엔드
- [Next.js 15.3.0](https://nextjs.org) - React 프레임워크
- [React 19](https://react.dev) - UI 라이브러리
- [Tailwind CSS](https://tailwindcss.com) - CSS 프레임워크

### 백엔드
- [FastAPI](https://fastapi.tiangolo.com) - 백엔드 API 프레임워크
- [Ultralytics YOLO v8](https://github.com/ultralytics/ultralytics) - 객체 탐지 모델
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - 광학 문자 인식
- [OpenCV](https://opencv.org) - 이미지 처리

## 🛠️ API 엔드포인트

- `GET /` - API 상태 확인
- `POST /analyze` - 이미지에서 의약품 감지
- `POST /analyze-pill` - 의약품 세부 분석 (색상, 모양, 텍스트 등)

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
