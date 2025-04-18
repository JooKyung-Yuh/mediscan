'use client';

import { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// 약품 분석 결과 타입 정의
interface PillAnalysisResult {
  color: string;
  texture: string;
  shape: string;
  text: string;
  image_no_bg?: string; // 배경 제거된 이미지의 base64 문자열
  drugName?: string;
  ingredients?: string;
  purpose?: string;
}

// 탐지 결과 타입 정의
interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class: number;
  class_name: string;
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(false);
  const [croppedImage, setCroppedImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<PillAnalysisResult | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
      // 새 이미지 업로드 시 기존 결과 초기화
      setDetections([]);
      setCroppedImage(null);
      setProcessedImage(null);
      setAnalysisResult(null);
    }
  };

  const handleAnalyze = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log(data);
      setDetections(data.detections);

      // 크롭된 이미지 생성
      if (data.detections.length > 0) {
        const img = new Image();
        img.src = preview!;
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          const detection = data.detections[0]; // 첫 번째 탐지된 객체 사용
          const originalWidth = img.width;
          const originalHeight = img.height;

          // 비율 계산
          const widthRatio = originalWidth / 640;
          const heightRatio = originalHeight / 640;

          // 좌표 변환
          const x1 = detection.x1 * widthRatio;
          const y1 = detection.y1 * heightRatio;
          const x2 = detection.x2 * widthRatio;
          const y2 = detection.y2 * heightRatio;

          const cropWidth = x2 - x1;
          const cropHeight = y2 - y1;

          canvas.width = cropWidth;
          canvas.height = cropHeight;

          ctx.drawImage(img, x1, y1, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
          const croppedDataUrl = canvas.toDataURL();
          setCroppedImage(croppedDataUrl);
          
          // 크롭이 완료되면 상세 분석 진행
          analyzePill(croppedDataUrl);
        };
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const analyzePill = async (croppedImageUrl: string) => {
    setAnalysisLoading(true);
    try {
      // 크롭된 이미지를 Blob으로 변환
      const response = await fetch(croppedImageUrl);
      const blob = await response.blob();
      
      // FormData 생성
      const formData = new FormData();
      formData.append('croppedImage', blob, 'cropped.jpg');
      
      // 상세 분석 API 호출
      const analysisResponse = await fetch('http://localhost:8000/analyze-pill', {
        method: 'POST',
        body: formData,
      });
      
      if (!analysisResponse.ok) {
        throw new Error(`API error: ${analysisResponse.status}`);
      }
      
      const analysisData = await analysisResponse.json();
      console.log('Analysis result:', analysisData);
      
      // 응답에 배경 제거된 이미지가 포함되어 있다면 설정
      if (analysisData.image_no_bg) {
        setProcessedImage(analysisData.image_no_bg);
      }
      
      setAnalysisResult(analysisData);
    } catch (error) {
      console.error('Error analyzing pill:', error);
      // 에러 발생 시에도 사용자에게 무언가 보여주기
      setAnalysisResult({
        color: '알 수 없음',
        texture: '알 수 없음',
        shape: '알 수 없음',
        text: '분석 실패'
      });
    } finally {
      setAnalysisLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle>MEDI SCAN</CardTitle>
          <CardDescription>알약 이미지를 업로드하여 약품 정보를 확인하세요</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col items-center space-y-4">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="image-upload"
            />
            <label
              htmlFor="image-upload"
              className="cursor-pointer border-2 border-dashed border-gray-300 rounded-lg p-4 w-full text-center"
            >
              {preview ? (
                <div className="relative">
                  <img 
                    src={preview} 
                    alt="Preview" 
                    className="max-h-64 mx-auto" 
                    id="preview-image"
                  />
                  {detections.map((detection, index) => (
                    <div
                      key={index}
                      style={{
                        position: 'absolute',
                        left: `${(detection.x1 / 640) * 100}%`,
                        top: `${(detection.y1 / 640) * 100}%`,
                        width: `${((detection.x2 - detection.x1) / 640) * 100}%`,
                        height: `${((detection.y2 - detection.y1) / 640) * 100}%`,
                        border: '2px solid red',
                        boxSizing: 'border-box',
                      }}
                    >
                      <div
                        style={{
                          position: 'absolute',
                          top: '-20px',
                          left: '0',
                          backgroundColor: 'red',
                          color: 'white',
                          padding: '2px 5px',
                          fontSize: '12px',
                        }}
                      >
                        {detection.class_name} ({Math.round(detection.confidence * 100)}%)
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p>이미지를 업로드하세요</p>
              )}
            </label>
            <Button 
              onClick={handleAnalyze} 
              disabled={!image || loading}
              className="w-full"
            >
              {loading ? '분석 중...' : '분석하기'}
            </Button>
            
            {croppedImage && (
              <div className="w-full border-t pt-4 mt-4">
                <h2 className="text-xl font-semibold mb-2">크롭된 이미지</h2>
                <div className="flex justify-center mb-4">
                  <img src={croppedImage} alt="Cropped" className="max-h-48" />
                </div>
                
                {processedImage && (
                  <div className="mt-4">
                    <h2 className="text-xl font-semibold mb-2">배경 제거된 이미지</h2>
                    <div className="flex justify-center mb-4 p-2 bg-gray-100 rounded">
                      <img 
                        src={processedImage} 
                        alt="No Background" 
                        className="w-auto max-w-full" 
                        style={{ maxHeight: '300px', objectFit: 'contain' }} 
                      />
                    </div>
                  </div>
                )}
                
                {analysisLoading ? (
                  <div className="text-center py-4">
                    <p>알약 상세 분석 중...</p>
                    <div className="mt-2 animate-pulse flex justify-center">
                      <div className="w-2 h-2 bg-blue-600 rounded-full mx-1"></div>
                      <div className="w-2 h-2 bg-blue-600 rounded-full mx-1 animate-delay-200"></div>
                      <div className="w-2 h-2 bg-blue-600 rounded-full mx-1 animate-delay-400"></div>
                    </div>
                  </div>
                ) : analysisResult ? (
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-2">분석 결과</h3>
                    <div className="grid grid-cols-2 gap-2 mb-4">
                      <div>
                        <span className="font-medium">색상:</span> 
                        <Badge variant="outline" className="ml-2">{analysisResult.color}</Badge>
                      </div>
                      <div>
                        <span className="font-medium">재질:</span> 
                        <Badge variant="outline" className="ml-2">{analysisResult.texture}</Badge>
                      </div>
                      <div>
                        <span className="font-medium">모양:</span> 
                        <Badge variant="outline" className="ml-2">{analysisResult.shape}</Badge>
                      </div>
                      <div>
                        <span className="font-medium">각인:</span> 
                        <Badge variant="outline" className="ml-2">{analysisResult.text}</Badge>
                      </div>
                    </div>
                    
                    {analysisResult.drugName ? (
                      <div className="border-t pt-3 mt-3">
                        <h4 className="font-semibold mb-2">약품 정보</h4>
                        <div className="space-y-1">
                          <p><span className="font-medium">약품명:</span> {analysisResult.drugName}</p>
                          <p><span className="font-medium">성분:</span> {analysisResult.ingredients}</p>
                          <p><span className="font-medium">용도:</span> {analysisResult.purpose}</p>
                        </div>
                      </div>
                    ) : (
                      <div className="border-t pt-3 mt-3 text-center text-gray-500">
                        <p>일치하는 약품 정보를 찾을 수 없습니다.</p>
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </main>
  );
}
