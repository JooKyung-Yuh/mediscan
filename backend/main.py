from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import io
import torch
import logging
import easyocr
from sklearn.cluster import KMeans
import os
import contextlib
import base64

# 배경 제거 모듈 임포트
from remove_bg import remove_background, remove_background_from_numpy

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

# PyTorch 2.6에서 weights_only 옵션을 False로 설정하기 위한 컨텍스트 매니저
@contextlib.contextmanager
def torch_load_with_weights_only_false():
    """PyTorch 2.6에서 weights_only=False로 torch.load를 사용하기 위한 컨텍스트 매니저"""
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        # weights_only 파라미터 추가 (2.6에서 기본값이 True로 변경됨)
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    # 함수 교체
    torch.load = patched_load
    try:
        yield
    finally:
        # 원래 함수 복원
        torch.load = original_load

# YOLO 모델 로드 시도
try:
    logger.info("Attempting to load YOLO model with weights_only=False...")
    
    # 안전한 클래스 추가 시도
    try:
        import torch.serialization
        from ultralytics import YOLO
        import ultralytics.nn.modules
        logger.info("Adding safe globals for YOLO model...")
        for module_name in dir(ultralytics.nn.modules):
            if not module_name.startswith('__'):
                try:
                    module = getattr(ultralytics.nn.modules, module_name)
                    if isinstance(module, type):
                        torch.serialization.add_safe_globals([module])
                except Exception as e:
                    logger.warning(f"Failed to add {module_name} to safe globals: {str(e)}")
    except Exception as e:
        logger.warning(f"Failed to setup safe globals: {str(e)}")
    
    # PyTorch 2.6 컨텍스트 매니저 사용
    with torch_load_with_weights_only_false():
        # YOLO 모델 로드
        yolo_model = YOLO('yolov8n.pt')
    
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    # 테스트를 위해 YOLO 로드 실패해도 계속 실행
    yolo_model = None
    logger.warning("Continuing without YOLO model for testing purposes")

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
        
        if yolo_model is None:
            # YOLO 모델 없이 더미 데이터 반환
            logger.warning("Using dummy detection data as YOLO model is not available")
            detections = [{
                "x1": 100,
                "y1": 100,
                "x2": 300,
                "y2": 300,
                "confidence": 0.95,
                "class": 0,
                "class_name": "pill"
            }]
            return {"detections": detections}
        
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
        
        # 가장 큰 바운딩 박스만 선택 (여러 개 감지된 경우)
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
        
        # 원본 이미지 크기 저장
        original_height, original_width = img.shape[:2]
        logger.info(f"Original image dimensions: {original_width}x{original_height}")
        
        # 배경 제거 수행
        logger.info("Removing background from cropped pill image")
        try:
            img_no_bg = await remove_background_from_numpy(img)
            logger.info("Background removal completed")
            
            # 배경 제거 후 크기 비교
            bg_removed_height, bg_removed_width = img_no_bg.shape[:2]
            logger.info(f"Background removed image dimensions: {bg_removed_width}x{bg_removed_height}")
            
            # 크기가 변경되었는지 확인
            if original_height != bg_removed_height or original_width != bg_removed_width:
                logger.warning(f"Size changed after background removal. Resizing back to original dimensions.")
                img_no_bg = cv2.resize(img_no_bg, (original_width, original_height))
                logger.info(f"Resized back to original dimensions: {original_width}x{original_height}")
            
            # 배경이 제거된 이미지를 base64로 인코딩하여 전송
            # BGRA -> RGBA로 변환
            if img_no_bg.shape[2] == 4:
                img_rgba = cv2.cvtColor(img_no_bg, cv2.COLOR_BGRA2RGBA)
            else:
                img_rgba = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2RGBA)
            
            # PIL 이미지로 변환
            pil_img = Image.fromarray(img_rgba)
            
            # 이미지를 base64로 인코딩
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_base64 = f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
            
        except Exception as e:
            logger.warning(f"Background removal failed: {str(e)}, continuing with original image")
            img_no_bg = img  # 실패 시 원본 이미지 사용
            img_base64 = None
        
        # 분석을 위해 복사본 생성
        img_for_analysis = img_no_bg.copy()
        
        # 이미지 전처리 (크기 조정, 노이즈 제거) - 분석용 이미지만 리사이즈
        img_for_analysis = cv2.resize(img_for_analysis, (300, 300))  # 분석을 위한 크기 조정
        
        # BGRA → RGB로 변환 (배경 제거 결과는 일반적으로 알파 채널이 있음)
        if img_for_analysis.shape[2] == 4:  # 알파 채널이 있는 경우
            img_rgb = cv2.cvtColor(img_for_analysis, cv2.COLOR_BGRA2RGB)
        else:
            img_rgb = cv2.cvtColor(img_for_analysis, cv2.COLOR_BGR2RGB)
            
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
            "text": text,
            "image_no_bg": img_base64  # 배경이 제거된 이미지 추가
        }
        
        # 약품 정보가 있으면 추가
        if pill_info:
            result.update(pill_info)
        
        logger.info(f"Analysis result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during pill analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-background")
async def remove_bg_endpoint(image: UploadFile = File(...)):
    """
    이미지의 배경을 제거하는 엔드포인트
    
    Args:
        image (UploadFile): 배경을 제거할 이미지 파일
    
    Returns:
        StreamingResponse: 배경이 제거된 이미지
    """
    try:
        logger.info("Received image for background removal")
        
        # 배경 제거 함수 호출
        output_buffer = await remove_background(image)
        
        # 스트리밍 응답으로 반환
        return StreamingResponse(
            output_buffer, 
            media_type="image/png"
        )
        
    except Exception as e:
        logger.error(f"Error during background removal API call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def analyze_color(img):
    """알약의 색상 분석 - 여러 색상 감지 지원"""
    try:
        # HSV 색상 공간으로 변환 (색상 인식에 더 적합)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_reshaped = img_hsv.reshape((-1, 3))
        
        # 여러 클러스터 사용 (여러 색상 추출)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(img_reshaped)
        
        # 각 클러스터의 크기(픽셀 수) 계산
        unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        
        # 클러스터를 크기 순으로 정렬 (내림차순)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_indices]
        sorted_labels = unique_labels[sorted_indices]
        
        # 전체 픽셀 수
        total_pixels = np.sum(counts)
        
        # 색상 이름과 비율을 저장할 리스트
        colors = []
        
        # 색상 이름 사전
        color_names = {
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
        
        # 중요한 클러스터만 처리 (5% 이상 차지하는 색상)
        for i, label in enumerate(sorted_labels):
            # 해당 클러스터가 전체의 5% 미만이면 무시
            if sorted_counts[i] / total_pixels < 0.05:
                continue
                
            # HSV 색상 가져오기
            color_hsv = kmeans.cluster_centers_[label]
            
            # HSV에서 RGB로 변환
            color_rgb = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
            r, g, b = color_rgb
            
            # 노란색 특별 처리 (HSV에서 더 정확히 판별 가능)
            h, s, v = color_hsv
            color_name = None
            
            if (20 <= h <= 60) and s > 50:  # 노란색 범위
                color_name = "노란색"
            else:
                # 가장 가까운 색상 찾기
                min_distance = float('inf')
                
                for name, (cr, cg, cb) in color_names.items():
                    distance = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        color_name = name
            
            # 색상과 점유율(%) 추가
            percentage = round((sorted_counts[i] / total_pixels) * 100)
            colors.append((color_name, percentage))
        
        # 결과 형식화
        if not colors:
            return "알 수 없음"
        elif len(colors) == 1:
            return colors[0][0]  # 단일 색상인 경우 색상명만 반환
        else:
            # 여러 색상인 경우 "주 색상(xx%), 부 색상(xx%)" 형식으로 반환
            main_color, main_pct = colors[0]
            secondary_color, sec_pct = colors[1]
            return f"{main_color}({main_pct}%), {secondary_color}({sec_pct}%)"
            
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
        
        # 이미지 전처리 - 각인이 잘 보이도록 대비 향상
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.equalizeHist(gray)
        
        # OCR 수행
        results = reader.readtext(enhanced)
        
        # 결과 처리
        if results:
            texts = [text for _, text, confidence in results if confidence > 0.3]
            if texts:
                return ' '.join(texts)
        
        return "각인 없음"
    except Exception as e:
        logger.error(f"Error in OCR analysis: {str(e)}")
        return "OCR 분석 오류"

def get_pill_info(color: str, shape: str, text: str) -> Optional[Dict[str, str]]:
    """
    알약 정보 검색 (모의 데이터)
    
    Args:
        color: 알약 색상
        shape: 알약 모양
        text: 알약 각인
    
    Returns:
        Optional[Dict[str, str]]: 알약 정보 또는 None
    """
    # 모의 데이터 - 실제로는 데이터베이스 조회가 필요
    pill_database = {
        ("흰색", "타원형", "M"): {
            "name": "아세트아미노펜",
            "usage": "해열, 진통제",
            "company": "약품제약",
            "side_effects": "두통, 어지러움, 구역질"
        },
        ("노란색", "원형", "R"): {
            "name": "이부프로펜",
            "usage": "소염진통제",
            "company": "건강약품",
            "side_effects": "위장장애, 졸음"
        }
    }
    
    # 간단한 검색 로직 (실제로는 더 복잡한 매칭 알고리즘 필요)
    for (db_color, db_shape, db_text), info in pill_database.items():
        if color == db_color and shape == db_shape and db_text in text:
            return info
    
    # 매칭되는 정보가 없는 경우
    return None

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 