import cv2
import numpy as np
import logging
from skimage import exposure
import imutils
import time

class ImageEnhancer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.logger = logging.getLogger(__name__)
        self.debug_timing = False
        
    def adaptive_histogram_equalization(self, image, clip_limit=2.0, tile_size=(8,8)):
        """Applying CLAHE with conservative parameters"""
        try:
            if len(image.shape) == 3:
                # Converting to LAB and enhance L channel only
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
                l_enhanced = clahe.apply(l)
                
                # Merge and convert back
                enhanced = cv2.merge((l_enhanced, a, b))
                return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
                return clahe.apply(image)
        except Exception as e:
            self.logger.error(f"Error in histogram equalization: {str(e)}")
            return image
    
    def single_scale_retinex(self, image, sigma=80):
        """Single Scale Retinex - simpler and more stable"""
        try:
            # Working with float32 for better precision
            img_float = image.astype(np.float32) + 1.0
            
            # Creating Gaussian blur
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
            
            # SSR formula: log(I) - log(blur(I))
            retinex = np.log(img_float) - np.log(blur + 1.0)
            
            # Normalize to 0-255
            retinex_normalized = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
            
            return retinex_normalized.astype(np.uint8)
        except Exception as e:
            self.logger.error(f"Error in SSR: {str(e)}")
            return image
    
    def multi_scale_retinex(self, image, scales=[15, 80, 200]):
        """Improved Multi-Scale Retinex"""
        try:
            if len(image.shape) == 3:
                # Process each channel separately
                channels = cv2.split(image)
                enhanced_channels = []
                
                for channel in channels:
                    channel_float = channel.astype(np.float32) + 1.0
                    retinex = np.zeros_like(channel_float)
                    
                    for sigma in scales:
                        blur = cv2.GaussianBlur(channel_float, (0, 0), sigma)
                        retinex += np.log(channel_float) - np.log(blur + 1.0)
                    
                    retinex = retinex / len(scales)
                    
                    # Normalize
                    retinex_norm = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
                    enhanced_channels.append(retinex_norm.astype(np.uint8))
                
                return cv2.merge(enhanced_channels)
            else:
                return self.single_scale_retinex(image)
                
        except Exception as e:
            self.logger.error(f"Error in MSR: {str(e)}")
            return image
    
    def enhance_for_face_detection(self, image, method='clahe'):
        """
        Enhanced preprocessing specifically for face detection
        
        Args:
            image: Input image
            method: 'clahe', 'retinex', or 'combined'
        """
        try:
            
            # Resize if too large
            if max(image.shape[:2]) > 1000:
                image = imutils.resize(image, width=1000)
            
            if method == 'clahe':
                # Conservative CLAHE - often best for face detection
                enhanced = self.adaptive_histogram_equalization(image, clip_limit=2.0)
                
            elif method == 'retinex':
                # Retinex enhancement
                enhanced = self.single_scale_retinex(image, sigma=80)
                
            elif method == 'combined':
                # First applying gentle CLAHE, then light Retinex
                clahe_enhanced = self.adaptive_histogram_equalization(image, clip_limit=1.5)
                
                # Convert to grayscale for face detection if needed
                # if len(clahe_enhanced.shape) == 3:
                #     # gray = cv2.cvtColor(clahe_enhanced, cv2.COLOR_BGR2GRAY)
                #     enhanced = self.single_scale_retinex(gray, sigma=80)
                # else:

                enhanced = self.single_scale_retinex(clahe_enhanced, sigma=80)
            
            else:
                enhanced = image
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error in enhancement: {str(e)}")
            return image
    
    def preprocess_image(self, image, enhancement_method='combined'):
        """Complete pipeline: enhance and detect faces"""
        try:
            t0 = time.time()
            # Load image
            # image = cv2.imread(image)
            # if image is None:
            #     print(f"Could not load image: {image}")
            #     return None, []
            
            # print(f"Original image shape: {image.shape}")
            print("Preprocessing and enhancing Frame...")
            # Enhance image
            enhanced = self.enhance_for_face_detection(image, method=enhancement_method)
            t1 = time.time()
            if self.debug_timing:
                self.logger.info(f"Enhancement ({enhancement_method}) took {(t1 - t0)*1000:.1f} ms")
            
            # cv2.imwrite("enhn.jpg", enhanced)
            print("Frame Enhanced!!!")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing Image: {str(e)}")
            return None, []

# Usage example
# if __name__ == "__main__":
#     enhancer = ImageEnhancer()
    
#     # Test different methods
#     methods = ['clahe', 'retinex', 'combined']
    
#     for method in methods:
#         print(f"\n--- Testing {method.upper()} method ---")
#         img = cv2.imread("./images/pic.jpg")
#         enhanced = enhancer.preprocess_image(img, "combined")

#         cv2.imwrite("result.jpg", enhanced)
#         # if faces is not None:
#         #     print(f"Method: {method}, Faces detected: {len(faces)}")
#         # else:
#         #     print(f"Method: {method} failed")
#     print("Done...")