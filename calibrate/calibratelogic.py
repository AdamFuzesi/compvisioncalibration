import numpy as np
import cv2
import glob
import os
from typing import List, Tuple, Optional, Dict
from pathlib import Path

"""
Author: Adam Fuzesi
Date: 2025/09/27

Program Layout description:

- Class based implementation for both Zhangs implementation and its visuals
as well as the cv2.calibratecamera implementation without its visual but simply its given parsed data

- Note I am not using given that I'm just comparing the calibration results with the manual implementation.

"""

class CalibrationConfig:
    # establishing basic geometric params 
    def __init__(self, checkerboardSize: Tuple[int, int] = (13, 9), squareSize: float = 20.0, outputDir: str = "output"):
        self.checkerboardSize = checkerboardSize
        self.squareSize = squareSize
        self.outputDir = Path(outputDir)
        self.corner_detection_dir = self.outputDir / "corner_detections"
        # creating output directories for the given program run instances
        self.outputDir.mkdir(exist_ok=True)
        self.corner_detection_dir.mkdir(exist_ok=True)


class CornerDetector:
    # handles logic for corner detection, reworked a lot and played around with images to various results and tweak them from there.
    def __init__(self, config: CalibrationConfig):
        self.config = config

        self.detection_flags = [
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH,
            None
        ]
        self.scales = [1.0, 0.75, 0.5, 0.25]
    
    def detectInImage(self, image_path: str) -> Tuple[bool, Optional[np.ndarray]]:
        # corner detection pre processing
        img = cv2.imread(image_path)
        if img is None:
            return False, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_gray = gray.copy()
        
        for scale in self.scales:
            # trying different scale portion combinatioms to identify
            if scale != 1.0:
                h, w = original_gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                gray = cv2.resize(original_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                gray = original_gray
            for flags in self.detection_flags:
                try:
                    if flags is not None:
                        ret, corners = cv2.findChessboardCorners(gray, self.config.checkerboardSize, flags )
                    else:
                        ret, corners = cv2.findChessboardCorners(gray, self.config.checkerboardSize)
                    
                    if ret:
                        # backset if needed
                        if scale != 1.0:
                            corners = corners / scale
                        # refine corner accuracy
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_refined = cv2.cornerSubPix(original_gray, corners, (11, 11), (-1, -1), criteria)
                        return True, corners_refined
                        
                except Exception:
                    continue
        
        return False, None
    
    def saveVisualization(self, image_path: str, corners: np.ndarray, image_idx: int) -> None:
        # simple function to save detected corners in a separate output folder to visualize, main info i want though is terminal printed data for the zhang implementation
        img = cv2.imread(image_path)
        if img is None:
            return
        # draws detected corners
        img_vis = img.copy()
        cv2.drawChessboardCorners(img_vis, self.config.checkerboardSize, corners, True)
        
        # resize if too large for better file size
        h, w = img_vis.shape[:2]
        if w > 1200:
            scale = 1200 / w
            img_vis = cv2.resize(img_vis, None, fx=scale, fy=scale)
        
        # saves to an output folder
        output_path = self.config.corner_detection_dir / f"corners_detected_image_{image_idx + 1:02d}.jpg"
        cv2.imwrite(str(output_path), img_vis)
        print(f"saved corner detection visualization to {output_path}")


class HomographyEstimator:
    # estimates the homography matrices using Direct Linear Transform.
    @staticmethod
    def estimate(worldP: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
        # trying to estimate homography matrix from differing world points to image points using dlt trans
        n_points = len(worldP)
        A = np.zeros((2 * n_points, 9))
        
        for i in range(n_points):
            X, Y = worldP[i, 0], worldP[i, 1]
            u, v = image_pts[i, 0], image_pts[i, 1]
            # builds matrix a for homogeneous linear system process
            A[2*i] = [-X, -Y, -1, 0, 0, 0, u*X, u*Y, u]
            A[2*i + 1] = [0, 0, 0, -X, -Y, -1, v*X, v*Y, v]
        # solves using svd methodology 
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1, :]
        H = h.reshape(3, 3)
        # normalizes so that h[2, 2] iss 1
        return H / H[2, 2]


class IntrinsicEstimator:
    # functionn estimates camera intrinsic parameters from homographies.
    @staticmethod
    def estimateOffHomographies(homographies: List[np.ndarray] ) -> np.ndarray:
        # get intrinsic matrix K from multiple homographies using zhang's method.
        def get_v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
            h_i, h_j = H[:, i], H[:, j]
            # retrieval for constraint equations
            return np.array([
                h_i[0] * h_j[0],
                h_i[0] * h_j[1] + h_i[1] * h_j[0],
                h_i[1] * h_j[1],
                h_i[2] * h_j[0] + h_i[0] * h_j[2],
                h_i[2] * h_j[1] + h_i[1] * h_j[2],
                h_i[2] * h_j[2]
            ])
        
        # builds constraint matrix v
        V = []
        for H in homographies:
            v12 = get_v_ij(H, 0, 1)
            v11 = get_v_ij(H, 0, 0)
            v22 = get_v_ij(H, 1, 1)
            
            V.append(v12)
            V.append(v11 - v22)
        
        V = np.array(V)
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1, :]
        # revert vt
        # must get intrinsic parameters from b listouts
        B11, B12, B22, B13, B23, B33 = b
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
        # then recover
        lambdaBasis = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
        
        alpha = np.sqrt(lambdaBasis / B11)
        beta = np.sqrt(lambdaBasis * B11 / (B11 * B22 - B12**2))
        gamma = -B12 * alpha**2 * beta / lambdaBasis
        u0 = gamma * v0 / beta - B13 * alpha**2 / lambdaBasis
        # and boom voila MATRIX
        return np.array([
            [alpha, gamma, u0],
            [0, beta, v0],
            [0, 0, 1]
        ])


class ExtrinsicEstimator:
    # function to estimate extrinsic params for each view
    @staticmethod
    def estimate(H: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # rotate estimates
        K_inv = np.linalg.inv(K)
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        # scale factor
        lambda1 = 1.0 / np.linalg.norm(K_inv @ h1)
        lambda2 = 1.0 / np.linalg.norm(K_inv @ h2)
        lambdaBasis = (lambda1 + lambda2) / 2.0
        # extracts the rotation columns and translation
        r1 = lambdaBasis * (K_inv @ h1)
        r2 = lambdaBasis * (K_inv @ h2)
        r3 = np.cross(r1, r2)
        t = lambdaBasis * (K_inv @ h3)
        # construct rotation matrix and enforce orthogonality using SVD
        R = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        return R, t

class ZhangCalibrationLogic:
    # zhangs implementation basis from scratch
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.corner_detector = CornerDetector(config)
        self.world_points = self._generate_world_points()
        
        # storage for calibration data
        self.image_points_list = []
        self.world_points_list = []
        self.image_size = None
        self.homographies = []
        # results
        self.K = None
        self.rvecs = []
        self.tvecs = []
    
    def _generate_world_points(self) -> np.ndarray:
        """Generate 3D coordinates of checkerboard corners in world frame."""
        cols, rows = self.config.checkerboardSize
        world_points = np.zeros((rows * cols, 3), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                world_points[i * cols + j] = [
                    j * self.config.squareSize,
                    i * self.config.squareSize,
                    0.0  
                    # planar grid
                ]
        return world_points
    
    def detectCorners(self, imagePaths: List[str], save_visualizations: bool = True) -> int:
        print(f"Detecting corners in {len(imagePaths)} images...")
        # principle logic for detecting corners in all the images
        print(f"Looking for checkerboard with {self.config.checkerboardSize[0]}x{self.config.checkerboardSize[1]} inner corners")
        
        successful_detections = 0
        
        for idx, image_path in enumerate(imagePaths):
            success, corners = self.corner_detector.detectInImage(image_path)
            
            if success:
                # Set image size from first successful detection
                if self.image_size is None:
                    img = cv2.imread(image_path)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.image_size = gray.shape[::-1]
                
                # Store detection results
                self.image_points_list.append(corners.reshape(-1, 2))
                self.world_points_list.append(self.world_points.copy())
                successful_detections += 1
                
                print(f"image {idx + 1}/{len(imagePaths)}: corners detected ✓")
                
                # Save visualization if requested
                if save_visualizations:
                    self.corner_detector.saveVisualization(image_path, corners, idx)
            else:
                print(f"image {idx + 1}/{len(imagePaths)}: corner detection failed ✗")
        
        print(f"\nSuccessfully detected corners in {successful_detections} images")
        return successful_detections
    
    def calibrate(self) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Perform Zhang's calibration algorithm."""
        if len(self.image_points_list) < 3:
            raise ValueError("Need at least 3 images with detected corners for calibration")
        
        print("\n--------- Zhang's Calibration ---------")
        #steps in the algo process, in order: step 1: estimate homographies, step 2: estimate intrinsic matrix, step 3: estimate extrinsic parameters
        print("\n" + "*"*60)
        print("estimating homographies...")
        print("*"*60)
        self.homographies = []
        for i, (worldP, image_pts) in enumerate(zip(self.world_points_list, self.image_points_list)):
            H = HomographyEstimator.estimate(worldP, image_pts)
            self.homographies.append(H)
            print(f"  View {i+1}: Homography estimated")
        
        print("\n" + "*"*60)
        print("estimating intrinsic matrix K...")
        print("*"*60)
        self.K = IntrinsicEstimator.estimateOffHomographies(self.homographies)
        print("Intrinsic matrix K:")
        print(self.K)
        
        print("\n" + "*"*60)
        print("estimating extrinsic parameters...")
        print("*"*60)
        self.rvecs = []
        self.tvecs = []
        
        for i, H in enumerate(self.homographies):
            R, t = ExtrinsicEstimator.estimate(H, self.K)
            rvec, _ = cv2.Rodrigues(R)
            
            self.rvecs.append(rvec)
            self.tvecs.append(t)
            
            print(f"  View {i+1}: Extrinsics estimated")
            print(f"    Rotation vector (rvec):")
            print(f"      {rvec.flatten()}")
            print(f"    Translation vector (t):")
            print(f"      {t.flatten()}")
            print(f"    Rotation matrix (R):")
            print(f"      {R}")
            print()
        
        # computing the reprojection error
        mean_error = self.computeReprojectionError()
        print(f"\n mean reprojection error: {mean_error:.4f} pixels")
        return self.K, self.rvecs, self.tvecs
    
    def computeReprojectionError(self) -> float:
        # computes mean reprojection error across all views
        totalError = 0
        totalPoints = 0
        
        for i, (worldP, image_pts) in enumerate(zip(self.world_points_list, self.image_points_list)):
            projected_pts, _ = cv2.projectPoints(
                worldP, self.rvecs[i], self.tvecs[i], self.K, None
            )
            projected_pts = projected_pts.reshape(-1, 2)
            
            error = np.linalg.norm(image_pts - projected_pts, axis=1)
            totalError += np.sum(error)
            totalPoints += len(error)
        
        return totalError / totalPoints
    
    def getIntrinsicParameters(self) -> Dict[str, float]:
        if self.K is None:
            return {}
        # to return the intrinsic parameters as a dictionary
        
        return {
            'fx': self.K[0, 0],
            'fy': self.K[1, 1],
            'cx': self.K[0, 2],
            'cy': self.K[1, 2],
            'skew': self.K[0, 1]
        }


class OpenCVCalibration:
    """OpenCV calibration for comparison with Zhang's method."""
    
    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.world_points = self._generate_world_points()
    
    def _generate_world_points(self) -> np.ndarray:
        """Generate 3D world coordinates for checkerboard."""
        cols, rows = self.config.checkerboardSize
        world_points = np.zeros((rows * cols, 3), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                world_points[i * cols + j] = [j * self.config.squareSize, i * self.config.squareSize, 0]
        return world_points
    
    def calibrate(self, imagePaths: List[str]) -> Dict[str, any]:
        """Perform OpenCV calibration."""
        print("\n--------- OpenCV Calibration --------- ")
        
        world_points_list = []
        image_points_list = []
        image_size = None
        
        # corner detect in all images
        for idx, image_path in enumerate(imagePaths):
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if image_size is None:
                image_size = gray.shape[::-1]
            
            ret, corners = cv2.findChessboardCorners(gray, self.config.checkerboardSize, None)
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                world_points_list.append(self.world_points)
                image_points_list.append(corners_refined)
        
        # Calibrate camera
        ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            world_points_list, image_points_list, image_size, None, None
        )
        
        print("Intrinsic matrix K:")
        print(K)
        print("\nDistortion coefficients:")
        print(dist_coeffs)
        
        print(f"\nextrinsic parameters obtained for {len(rvecs)} images:")
        print("rotation vectors:")
        for i, rvec in enumerate(rvecs):
            print(f"  Image {i+1}: {rvec.flatten()}")
        
        print("\ntranslation vectors:")
        for i, tvec in enumerate(tvecs):
            print(f"  Image {i+1}: {tvec.flatten()}")
        
        # getting the reprojection error
        totalError = 0
        for i in range(len(world_points_list)):
            projected_pts, _ = cv2.projectPoints(
                world_points_list[i], rvecs[i], tvecs[i], K, dist_coeffs
            )
            error = cv2.norm(image_points_list[i], projected_pts, cv2.NORM_L2) / len(projected_pts)
            totalError += error
        
        mean_error = totalError / len(world_points_list)
        print(f"\nMean reprojection error: {mean_error:.4f} pixels")
        # main stored result dictionnary
        return {
            'K': K,
            'dist_coeffs': dist_coeffs,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'reprojection_error': mean_error,
            'fx': K[0, 0],
            'fy': K[1, 1],
            'cx': K[0, 2],
            'cy': K[1, 2]
        }


class CalibrationComparator:
    # function to compare zhang's method with opencv calibrations resutls
    @staticmethod
    def compare(zhang_result: Dict[str, float], opencv_result: Dict[str, any]) -> None:
        # comparing zhang's method with open cv calibration results
        print("\n" + "*"*60)
        print("CALIBRATION COMPARISON")
        print("*"*60)
        
        print("\n{:<15} {:>20} {:>20} {:>10}".format("Parameter", "Zhang's Method", "OpenCV", "Diff %"))
        print("-" * 70)
        
        params = ['fx', 'fy', 'cx', 'cy']
        for param in params:
            zhang_val = zhang_result[param]
            opencv_val = opencv_result[param]
            diff_pct = abs(zhang_val - opencv_val) / opencv_val * 100
            print(f"{param:<15} {zhang_val:>20.4f} {opencv_val:>20.4f} {diff_pct:>9.2f}%")
        
        print("\n" + "*"*60)
        print(f"Zhang's skew parameter: {zhang_result.get('skew', 0):.6f}")
        print("*"*60)


class CheckerboardSizeDetector:
    """Auto-detect checkerboard size from images."""
    
    COMMON_SIZES = [
        (13, 9),   # 14x10 checkerboard
        (12, 8),   # 13x9 checkerboard
        (9, 6),    # 10x7 checkerboard
        (8, 5),    # 9x6 checkerboard
        (10, 7),   # 11x8 checkerboard
        (7, 5),    # 8x6 checkerboard
        (6, 4),    # 7x5 checkerboard
    ]
    
    @staticmethod
    def detect_size(image_path: str) -> Optional[Tuple[int, int]]:
        """Try to detect checkerboard size from an image."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"\nTrying to detect checkerboard in: {os.path.basename(image_path)}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        for size in CheckerboardSizeDetector.COMMON_SIZES:
            ret, _ = cv2.findChessboardCorners(
                gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                print(f"✓ Found checkerboard with {size[0]}x{size[1]} inner corners")
                return size
        
        print("✗ Could not detect any standard checkerboard size")
        return None


def findCalibrationImages(image_dir: str) -> List[str]:
    # simple start up logic to find adn load images
    extensions = ['*.jpg', '*.png']
    # had some png versions at first from screenshots and trial pics
    imagePaths = []
    for ext in extensions:
        imagePaths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(imagePaths)

def main():
    # main program run start up and process
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    image_dir = project_root / "images"
    # locating calibration images
    imagePaths = findCalibrationImages(str(image_dir))
    if not imagePaths:
        print(f"no images found in '{image_dir}' directory")
        return
    print(f"found {len(imagePaths)} calibration images in: {image_dir}")
    
    # detecting checkerboard size
    print("\n" + "*"*60)
    print("detecting the checkerboards size")
    print("*"*60)
    
    detectedSize = CheckerboardSizeDetector.detect_size(imagePaths[0])

    # NOTE: this is a default size if the auto detection fails, happened quite a bit in my initial approach, hardcoding in the size preemptively worked for the rest of the implementations but  auto detection works great now
    checkerboardSize = detectedSize if detectedSize else (13, 9)
    
    if detectedSize is None:
        print("\n auto detection failed. use given default config")
        print(f"setting checkerboard size to {checkerboardSize}")
    
    # sets in motion the configuration
    config = CalibrationConfig(
        checkerboardSize=checkerboardSize,
        squareSize=20.0,
        outputDir=str(project_root / "output")
    )
    print(f"\nusing checkerboard size: {config.checkerboardSize}")
    print(f"square size: {config.squareSize} mm")
    print(f"visuals shown in directory: {config.outputDir}")
    print("*"*60)
    
    # init zhang's method!!!!!!!!
    zhang = ZhangCalibrationLogic(config)
    n_detected = zhang.detectCorners(imagePaths, save_visualizations=True)
    
    if n_detected < 1:
        # in my initial approach the images I used were weirdly blurry and the lighting was off with shadows 
        print("\nerror, get at least 1 image with detected corners")
        return
    
    K_zhang, rvecs_zhang, tvecs_zhang = zhang.calibrate()
    zhang_params = zhang.getIntrinsicParameters()
    
    # using built in open cv method to compare results between my manual implementation... please refer to terminal output
    opencvCalibrator = OpenCVCalibration(config)
    opencv_params = opencvCalibrator.calibrate(imagePaths)
    CalibrationComparator.compare(zhang_params, opencv_params)
    
    print("\n calibration complete!")
    print(f"corner detection images saved to {config.corner_detection_dir}")
    print(f"\nzhang's Intrinsic Matrix K:")
    print(K_zhang)

if __name__ == "__main__":
    main()
