import React, { useState } from 'react';
import { Button } from './ui/button';
import { Loader2 } from 'lucide-react';

interface RemoveBgButtonProps {
  image: File | null;
  setProcessedImage: (url: string | null) => void;
}

const RemoveBgButton: React.FC<RemoveBgButtonProps> = ({ image, setProcessedImage }) => {
  const [loading, setLoading] = useState(false);

  const handleRemoveBg = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:8000/remove-background', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      // 응답이 이미지 데이터이므로 Blob으로 처리합니다
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setProcessedImage(imageUrl);
    } catch (error) {
      console.error('Error removing background:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Button 
      onClick={handleRemoveBg} 
      disabled={!image || loading}
      className="w-full mt-2"
      variant="outline"
    >
      {loading ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          배경 제거 중...
        </>
      ) : (
        '배경 제거하기'
      )}
    </Button>
  );
};

export default RemoveBgButton; 