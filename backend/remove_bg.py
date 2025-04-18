from fastapi import UploadFile, HTTPException
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

try:
    # rembg 라이브러리 임포트
    from rembg import remove, new_session
    # 세션 생성 (성능 향상을 위해 한 번만 생성)
    session = new_session()
    logger.info("rembg session initialized successfully")
except ImportError:
    logger.error("rembg library not found. Please install it with: pip install rembg")
    session = None

async def remove_background(image: UploadFile) -> BytesIO:
    """
    이미지의 배경을 제거하는 함수
    
    Args:
        image (UploadFile): 배경을 제거할 이미지 파일
    
    Returns:
        BytesIO: 배경이 제거된 이미지를 담은 BytesIO 객체
    
    Raises:
        HTTPException: 배경 제거 중 오류가 발생한 경우
    """
    try:
        logger.info("Processing image for background removal")
        
        if session is None:
            raise HTTPException(status_code=500, detail="rembg library not available")
        
        # 이미지 읽기
        contents = await image.read()
        
        # rembg를 사용하여 배경 제거 (세션 재사용)
        output = remove(contents, session=session)
        
        # 결과를 BytesIO에 저장
        output_buffer = BytesIO(output)
        output_buffer.seek(0)
        
        logger.info("Background removal completed successfully")
        return output_buffer
    
    except Exception as e:
        logger.error(f"Error during background removal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

async def remove_background_from_numpy(img: np.ndarray) -> np.ndarray:
    """
    NumPy 배열로부터 배경을 제거하는 함수
    
    Args:
        img (np.ndarray): 배경을 제거할 이미지의 NumPy 배열 (BGR 형식)
    
    Returns:
        np.ndarray: 배경이 제거된 이미지의 NumPy 배열 (BGRA 형식)
    
    Raises:
        HTTPException: 배경 제거 중 오류가 발생한 경우
    """
    try:
        logger.info("Processing numpy array for background removal")
        logger.info(f"Input image shape: {img.shape}")
        
        if session is None:
            raise HTTPException(status_code=500, detail="rembg library not available")
        
        # BGR에서 RGB로 변환 (rembg는 RGB 형식의 입력을 기대함)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.info(f"RGB converted image shape: {img_rgb.shape}")
        
        # rembg를 사용하여 배경 제거
        output = remove(img_rgb, session=session)
        logger.info(f"After rembg output shape: {output.shape}")
        
        # RGBA에서 BGRA로 변환
        output_bgra = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
        logger.info(f"Final BGRA output shape: {output_bgra.shape}")
        
        # 원본 크기와 결과 크기가 다른 경우 리사이즈
        if img.shape[:2] != output_bgra.shape[:2]:
            logger.warning(f"Size mismatch detected! Resizing from {output_bgra.shape[:2]} to {img.shape[:2]}")
            output_bgra = cv2.resize(output_bgra, (img.shape[1], img.shape[0]))
            logger.info(f"After resize output shape: {output_bgra.shape}")
        
        logger.info("Background removal from numpy array completed successfully")
        return output_bgra
    
    except Exception as e:
        logger.error(f"Error during background removal from numpy array: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Background removal from numpy array failed: {str(e)}") 