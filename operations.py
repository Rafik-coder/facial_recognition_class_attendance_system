import cv2
import numpy as np
from scipy.spatial import distance
import logging
from skimage import exposure
import imutils

class ImageEnhancer:
    def __init__(self):
        self.gamma = 1.8  # Reduced gamma for better low-light handling
        self.sigma = 125
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.logger = logging.getLogger(__name__)
        
    def enhance_low_light(self, image):
        """Enhanced low-light image processing"""
        try:
            # Check if image is grayscale
            if len(image.shape) == 2:  # Grayscale image
                # Apply CLAHE directly
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
                
                # Apply gamma correction
                enhanced = exposure.adjust_gamma(enhanced, gamma=0.8)
                
                # Apply additional contrast enhancement
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
                
                return enhanced
            else:  # Color image
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel with optimized parameters
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                
                # Apply gamma correction to L channel
                cl = exposure.adjust_gamma(cl, gamma=0.8)
                
                # Merge channels
                merged = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                
                # Apply additional contrast enhancement
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
                
                return enhanced
        except Exception as e:
            self.logger.error(f"Error in low-light enhancement: {str(e)}")
            return image

    def histogram_equalization(self, image):
        """Apply adaptive histogram equalization with improved parameters"""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel with optimized parameters
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                
                # Apply gamma correction to L channel
                cl = exposure.adjust_gamma(cl, gamma=0.8)
                
                # Merge channels
                merged = cv2.merge((cl, a, b))
                return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
                return exposure.adjust_gamma(enhanced, gamma=0.8)
        except Exception as e:
            self.logger.error(f"Error in histogram equalization: {str(e)}")
            return image
    
    def retinex_enhancement(self, image):
        """Enhanced Multi-Scale Retinex (MSR) implementation with optimized parameters"""
        try:
            scales = [15, 80, 200]
            img_float = image.astype(np.float64) + 1.0
            img_log = np.log(img_float)
            
            retinex = np.zeros_like(img_log)
            
            for scale in scales:
                blur = cv2.GaussianBlur(img_float, (0, 0), scale)
                blur_log = np.log(blur)
                retinex += img_log - blur_log
            
            retinex = retinex / len(scales)
            
            # Apply gamma correction with safety checks
            retinex = np.clip(retinex, 0, None)
            retinex = np.power(retinex, 1/self.gamma)
            
            # Normalize and convert to uint8
            retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
            return np.clip(retinex, 0, 255).astype(np.uint8)
        except Exception as e:
            self.logger.error(f"Error in retinex enhancement: {str(e)}")
            return image
    
    def preprocess_image(self, image):
        """Enhanced image preprocessing for face detection with low-light handling"""
        try:
            print("Preprocessing Image...")
            # Step 1: Resizing image
            if max(image.shape) > 1000:
                image = imutils.resize(image, width=1000)
            
            # Step 2: Applying low-light enhancement
            enhanced = self.enhance_low_light(image)
            
            # Step 3: Applying histogram equalization
            he_enhanced = self.histogram_equalization(enhanced)

            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_retinex = self.retinex_enhancement(l)   # Applying only to luminance
            merged = cv2.merge((l_retinex, a, b))
            ret_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

            
            # Step 4: Applying Retinex enhancement
            # if len(he_enhanced.shape) == 3:
            #     channels = cv2.split(he_enhanced)
            #     enhanced_channels = []
            #     for channel in channels:
            #         enhanced_channel = self.retinex_enhancement(channel)
            #         enhanced_channels.append(enhanced_channel)
            #     final_enhanced = cv2.merge(enhanced_channels)
            # else:
            #     final_enhanced = self.retinex_enhancement(he_enhanced)
            
            # ret_enhanced = self.retinex_enhancement(image)
            # Step 5: Apply additional contrast enhancement
            # final_enhanced = exposure.adjust_gamma(final_enhanced, gamma=1.2)

            print("savving..")
            # cv2.imwrite("./images/he_enhanced3.jpg", he_enhanced)
            # cv2.imwrite("./images/ret_enhanced2.jpg", ret_enhanced)
            # cv2.imwrite("./images/final_enhanced3.jpg", final_enhanced)
            
            return enhanced
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {str(e)}")
            return image


# image = cv2.imread("./images/pic.jpg")
# # image = cv2.imread("./images/test.png")
# tester = ImageEnhancer()
# tester.preprocess_image(image)