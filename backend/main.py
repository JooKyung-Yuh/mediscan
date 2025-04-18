from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import io
import torch
import logging
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
import easyocr
from sklearn.cluster import KMeans

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# 안전한 글로벌 설정
torch.serialization.add_safe_globals([Sequential])
torch.serialization.add_safe_globals([Conv])
torch.serialization.add_safe_globals([C2f])
torch.serialization.add_safe_globals([SPPF])
torch.serialization.add_safe_globals([Detect])
torch.serialization.add_safe_globals([DetectionModel])

# YOLO 모델 로드
try:
    logger.info("Loading YOLO model...")
    yolo_model = YOLO('yolov8n.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise HTTPException(status_code=500, detail="Failed to load YOLO model")

# OCR 리더 초기화
try:
    logger.info("Initializing OCR reader...")
    reader = easyocr.Reader(['en', 'ko'])
    logger.info("OCR reader initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OCR reader: {str(e)}")
    reader = None  # OCR이 실패해도 다른 기능은 작동하도록 함

@app.get("/")
async def root():
    return {"message": "MEDI SCAN API"}

@app.post("/analyze")
async def analyze_pill(image: UploadFile = File(...)):
    try:
        logger.info("Received image for analysis")
        
        # 이미지 읽기
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # 이미지 전처리
        img = cv2.resize(img, (640, 640))  # YOLO 모델에 맞는 크기로 조정
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        
        # YOLO 모델로 알약 탐지
        results = yolo_model(img, conf=0.1)  # confidence threshold 낮춤
        
        # 탐지된 객체 정보 추출
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])]  # 클래스 이름 추가
                })
        
        # 가장 큰 바운딩 박스만 선택
        if detections:
            areas = [(d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) for d in detections]
            max_area_idx = areas.index(max(areas))
            detections = [detections[max_area_idx]]
        
        logger.info(f"Detected {len(detections)} objects")
        return {"detections": detections}
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-pill")
async def analyze_pill_details(croppedImage: UploadFile = File(...)):
    try:
        logger.info("Received cropped image for detailed analysis")
        
        # 이미지 읽기
        contents = await croppedImage.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # 이미지 전처리 (크기 조정, 노이즈 제거)
        img = cv2.resize(img, (300, 300))  # 분석을 위한 크기 조정
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
        
        # 1. 색상 분석
        color = analyze_color(img_denoised)
        
        # 2. 재질 분석
        texture = analyze_texture(img_denoised)
        
        # 3. 모양 분석
        shape = analyze_shape(img_denoised)
        
        # 4. 글씨 인식 (OCR)
        text = analyze_text(img_denoised)
        
        # 5. 알약 데이터베이스 조회 (여기서는 모의 데이터 반환)
        pill_info = get_pill_info(color, shape, text)
        
        result = {
            "color": color,
            "texture": texture,
            "shape": shape,
            "text": text
        }
        
        # 약품 정보가 있으면 추가
        if pill_info:
            result.update(pill_info)
        
        logger.info(f"Analysis result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during pill analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def analyze_color(img):
    """알약의 주 색상 분석 개선"""
    try:
        # HSV 색상 공간으로 변환 (색상 인식에 더 적합)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_reshaped = img_hsv.reshape((-1, 3))
        
        # 여러 클러스터 사용 (더 다양한 색상 추출)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(img_reshaped)
        
        # 가장 많이 나타나는 색상 찾기
        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster = unique_labels[np.argmax(counts)]
        dominant_color_hsv = kmeans.cluster_centers_[dominant_cluster]
        
        # HSV에서 RGB로 변환
        dominant_color_rgb = cv2.cvtColor(np.uint8([[dominant_color_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
        r, g, b = dominant_color_rgb
        
        # 노란색 판별 개선 (HSV에서 더 정확히 판별 가능)
        h, s, v = dominant_color_hsv
        if (20 <= h <= 60) and s > 50:  # 노란색 범위
            return "노란색"
        
        # 이전 색상 판별 로직
        colors = {
            '빨간색': (255, 0, 0),
            '초록색': (0, 255, 0),
            '파란색': (0, 0, 255),
            '노란색': (255, 255, 0),
            '자주색': (128, 0, 128),
            '오렌지색': (255, 165, 0),
            '분홍색': (255, 192, 203),
            '갈색': (165, 42, 42),
            '검정색': (0, 0, 0),
            '흰색': (255, 255, 255),
            '회색': (128, 128, 128)
        }
        
        min_distance = float('inf')
        color_name = '알 수 없음'
        
        for name, (cr, cg, cb) in colors.items():
            distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                color_name = name
        
        return color_name
    except Exception as e:
        logger.error(f"Error in color analysis: {str(e)}")
        return "알 수 없음"

def analyze_texture(img):
    """알약의 재질 분석"""
    try:
        # 그레이스케일로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 텍스처 특성 계산 (예: GLCM, Haralick 텍스처 등)
        # 여기서는 단순화를 위해 표준 편차를 사용
        std_dev = np.std(gray)
        
        # 표준 편차에 따른 재질 분류
        if std_dev < 15:
            return "매끄러운 표면"
        elif std_dev < 30:
            return "약간 거친 표면"
        else:
            return "거친 표면"
    except Exception as e:
        logger.error(f"Error in texture analysis: {str(e)}")
        return "알 수 없음"

def analyze_shape(img):
    """알약의 모양 분석"""
    try:
        # 그레이스케일로 변환
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 이진화
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return "알 수 없음"
        
        # 가장 큰 윤곽선 사용
        cnt = max(contours, key=cv2.contourArea)
        
        # 모양 특성 계산
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # 원형 정도 측정
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 모양 분류
        if circularity > 0.85:
            return "원형"
        elif circularity > 0.65:
            return "타원형"
        elif circularity > 0.4:
            return "캡슐형"
        else:
            return "불규칙한 형태"
    except Exception as e:
        logger.error(f"Error in shape analysis: {str(e)}")
        return "알 수 없음"

def analyze_text(img):
    """OCR을 통한 알약 각인 텍스트 인식 개선"""
    try:
        if reader is None:
            return "OCR 분석 불가"
        
        # 이미지 전처리 강화
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 다양한 전처리 방법 시도
        processed_images = [
            gray,  # 원본 그레이스케일
            cv2.equalizeHist(gray),  # 히스토그램 평활화
            cv2.GaussianBlur(gray, (5, 5), 0)  # 블러링
        ]
        
        # 여러 전처리 이미지에 대해 OCR 수행
        all_results = []
        for processed_img in processed_images:
            # 대비 개선
            alpha = 1.5  # 대비 조정 계수
            beta = 10    # 밝기 조정 계수
            adjusted = cv2.convertScaleAbs(processed_img, alpha=alpha, beta=beta)
            
            # OCR 수행
            results = reader.readtext(adjusted)
            if results:
                all_results.extend(results)
        
        if not all_results:
            return "각인 없음"
        
        # 신뢰도가 높은 결과만 선택
        confident_results = [result[1] for result in all_results if result[2] > 0.3]
        if confident_results:
            return ", ".join(confident_results)
        else:
            return "각인 인식 불확실"
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        return "분석 실패"

def get_pill_info(color: str, shape: str, text: str) -> Optional[Dict[str, str]]:
    """색상, 모양, 각인을 바탕으로 약품 정보 조회 (모의 데이터)"""
    # 실제로는 데이터베이스에서 조회해야 함
    # 모의 데이터
    mock_db = [
        {
            "color": "흰색",
            "shape": "원형",
            "text": "CP",
            "drugName": "시프로플록사신",
            "ingredients": "시프로플록사신 500mg",
            "purpose": "항생제"
        },
        {
            "color": "노란색",
            "shape": "원형",
            "text": "CP",
            "drugName": "시프로플록사신",
            "ingredients": "시프로플록사신 500mg",
            "purpose": "항생제"
        },
        {
            "color": "흰색",
            "shape": "타원형",
            "text": "A",
            "drugName": "아스피린",
            "ingredients": "아세틸살리실산 100mg",
            "purpose": "진통제, 해열제"
        }
    ]
    
    # 일치하는 약품 찾기
    for pill in mock_db:
        if (pill["color"] == color and 
            pill["shape"] == shape and 
            pill["text"] == text):
            return {
                "drugName": pill["drugName"],
                "ingredients": pill["ingredients"],
                "purpose": pill["purpose"]
            }
    
    # 일치하는 약품이 없으면 None 반환
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 