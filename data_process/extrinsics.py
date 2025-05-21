import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

points = {}
# 3D points in world coordinates
points["cam2"] = {}
points["cam2"]["2d"] = np.array([
    [304, 50],  # mark 4
    [502, 55],
    [813, 61],
    [1063, 64],
    [1308, 69],
    [1549, 70],
    [312, 309],  # mark 3
    [563, 311],
    [814, 313],
    [1060, 315],
    [1303, 317],
    [1542, 318],
    [318, 553],  # mark 2
    [570, 555],
    [816, 557],
    [1059, 559],
    [1300, 560]
], dtype=np.float32)

# Corresponding 2D points in image
points["cam2"]["3d"] = np.array([
    [100, 400, 0],
    [200, 400, 0],
    [300, 400, 0],
    [400, 400, 0],
    [500, 400, 0],
    [600, 400, 0],
    [100, 300, 0],
    [200, 300, 0],
    [300, 300, 0],
    [400, 300, 0],
    [500, 300, 0],
    [600, 300, 0],
    [100, 200, 0],
    [200, 200, 0],
    [300, 200, 0],
    [400, 200, 0],
    [500, 200, 0],
], dtype=np.float32)

points["cam4"] = {}
points["cam4"]["2d"] = np.array([
    [376, 125],  # mark 6
    [619, 136],
    [859, 147],
    [1100, 157],
    [1339, 169],
    [1578, 178],
    [361, 358],  # mark 5
    [606, 371],
    [847, 384],
    [1091, 397],
    [1332, 408],
    [1570, 417],
    [346, 599],  # mark 4
    [593, 612],
    [836, 625],
    [1080, 635],
    [1323, 649],
    [330, 851],  # mark 3
    [575, 862],
    [823, 872],
    [1069, 884],
    [1314, 895],
    [1558, 906],
], dtype=np.float32)

points["cam4"]["3d"] = np.array([
    [100, 600, 0],  # mark 6
    [200, 600, 0],
    [300, 600, 0],
    [400, 600, 0],
    [500, 600, 0],
    [600, 600, 0],
    [100, 500, 0],  # mark 5
    [200, 500, 0],
    [300, 500, 0],
    [400, 500, 0],
    [500, 500, 0],
    [600, 500, 0],
    [100, 400, 0],  # mark 4   
    [200, 400, 0],
    [300, 400, 0],
    [400, 400, 0],
    [500, 400, 0],
    [100, 300, 0],  # mark 3
    [200, 300, 0],
    [300, 300, 0],
    [400, 300, 0],
    [500, 300, 0],
    [600, 300, 0],
], dtype=np.float32)

points["cam6"] = {}
points["cam6"]["2d"] = np.array([
    [411, 211],  # mark 8
    [646, 218],
    [879, 227],
    [1118, 233],
    [1357, 240],
    [1600, 245],
    [396, 442],  # mark 7
    [633, 451],
    [871, 461],
    [1112, 467],
    [1355, 477],
    [1602, 482],
    [376, 688],  # mark 6
    [618, 697],
    [860, 704],
    [1106, 713],
    [1353, 722],
    [1602, 730],
    [356, 930],  # mark 5
    [603, 941],
    [849, 954],
    [1101, 966],
    [1353, 977],
], dtype=np.float32)

points["cam6"]["3d"] = np.array([
    [100, 800, 0],
    [200, 800, 0],
    [300, 800, 0],
    [400, 800, 0],
    [500, 800, 0],
    [600, 800, 0],
    [100, 700, 0],
    [200, 700, 0],
    [300, 700, 0],
    [400, 700, 0],
    [500, 700, 0],
    [600, 700, 0],
    [100, 600, 0],
    [200, 600, 0],
    [300, 600, 0],
    [400, 600, 0],
    [500, 600, 0],
    [600, 600, 0],
    [100, 500, 0],
    [200, 500, 0],
    [300, 500, 0],
    [400, 500, 0],
    [500, 500, 0],
], dtype=np.float32)

points["cam8"] = {}
points["cam8"]["2d"] = np.array([
    [401, 173],  # mark 10
    [639, 185],
    [876, 198],
    [1116, 208],
    [1357, 218],
    [1600, 228],
    [385, 408],  # mark 9
    [624, 421],
    [861, 434],
    [1105, 447],
    [1348, 458],
    [1593, 468],
    [365, 647],  # mark 8
    [609, 661],
    [850, 676],
    [1095, 689],
    [1340, 702],
    [1586, 713],    
    [348, 893],  # mark 7
    [593, 908],
    [838, 924],
    [1084, 937],
    [1330, 952],
    [1581, 962],
], dtype=np.float32)

points["cam8"]["3d"] = np.array([
    [100, 1000, 0],
    [200, 1000, 0],
    [300, 1000, 0],
    [400, 1000, 0],
    [500, 1000, 0],
    [600, 1000, 0],
    [100, 900, 0],
    [200, 900, 0],
    [300, 900, 0],
    [400, 900, 0],
    [500, 900, 0],
    [600, 900, 0],
    [100, 800, 0],
    [200, 800, 0],
    [300, 800, 0],
    [400, 800, 0],
    [500, 800, 0],
    [600, 800, 0],
    [100, 700, 0],
    [200, 700, 0],
    [300, 700, 0],
    [400, 700, 0],
    [500, 700, 0],
    [600, 700, 0],
], dtype=np.float32)

def get_extrinsics(cam_num, parent_folder, points_3d, points_2d):
    intrinsics_path = f"{parent_folder}/camera-calibration/cam{cam_num}/intrinsic.json"

    with open(intrinsics_path, 'r') as f:
        intrinsics = json.load(f)

    camera_matrix = np.array(intrinsics['intrinsic'])
    dist_coeffs = np.array(intrinsics['distortion_coefficients'])# Solve PnP
    success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs)

    # Convert rotation vector to matrix
    print(rvec)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec

def observe_video(vid_path, frame_number=100):
    # # Read the video
    cap = cv2.VideoCapture(vid_path)

    # Set frame number (you can change this to any frame number you want)

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Convert BGR to RGB (OpenCV uses BGR, matplotlib uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Plot the frame
        plt.figure(figsize=(10, 8))
        plt.imshow(frame_rgb)
        plt.title(f'Frame {frame_number}')
        plt.axis('off')
        plt.show()
    else:
        print(f"Could not read frame {frame_number}")

    # Release the video capture
    cap.release()


def main():
    # cam_num = 8
    video_names = {
        2: "GH010003.MP4",
        4: "GH020010_rot.MP4",
        6: "GH010162.MP4",
        8: "GH010165.MP4",
    }
    raw_folder = "/home/zonghuan/tudelft/projects/datasets/conflab/data_raw/cameras"
    processed_folder = "/home/zonghuan/tudelft/projects/datasets/conflab/data_processed/cameras"

    for cam_num in [2, 4, 6, 8]:
        cam_name = "cam" + str(cam_num)
        vid_path = f"{raw_folder}/video/cam0{cam_num}/{video_names[cam_num]}"
        # observe_video(vid_path, frame_number=2)
        R, tvec = get_extrinsics(cam_num, raw_folder, points[cam_name]["3d"], points[cam_name]["2d"])
        print(R)
        print(tvec)
        save_path = f"{processed_folder}/camera_calibration/cam{cam_num}/extrinsic_zh.json"
        # with open(save_path, 'w') as f:
        #     json.dump({"rotation": R.tolist(), "translation": tvec.tolist()}, f)


if __name__ == "__main__":
    main()
