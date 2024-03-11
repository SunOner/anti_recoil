import cv2
import win32gui, win32con, win32api
import numpy as np
from logic.capture import Config
cfg = Config()
from logic.capture import Capture
from logic.mouse import MouseThread

def spawn_debug_window():
    cv2.namedWindow('antirecoil')
    window_hwnd = win32gui.FindWindow(None, 'antirecoil')
    win32gui.SetWindowPos(window_hwnd, win32con.HWND_TOPMOST, 10, 10, 200, 200, 0)

def get_features(gray, mask):
    return cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

def create_mask(x, y, width, height, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y:y+height, x:x+width] = 255
    return mask

def init():
    old_gray = None
    p0 = None
    first_frame = True

    if debug_window:
        spawn_debug_window()
        
    mask_region = frames.Calculate_screen_offset(custom_region=(cfg.detection_window_width, cfg.detection_window_height))
    mask = create_mask(x=mask_region[0], y=int(mask_region[1]), width=int(mask_region[2]), height=int(mask_region[3] - 250), shape=(cfg.detection_window_height, cfg.detection_window_width))
    
    while True:
        frame = frames.get_new_frame()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mask_window:
            mask_vis = cv2.merge([mask, mask, mask])
            frame_with_mask = cv2.addWeighted(frame, 1, mask_vis, 0.3, 0)
        
        state = win32api.GetAsyncKeyState(2)
        if state < 0:
            if first_frame:
                p0 = get_features(frame_gray, mask)
                if p0 is not None:
                    for i in p0:
                        x, y = map(int, i.ravel())
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                old_gray = frame_gray.copy()
                first_frame = False
                continue
            
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

                if len(good_new) > 0:
                    movement_x = np.median(good_new[:, 0]) - np.median(good_old[:, 0])
                    movement_y = np.median(good_new[:, 1]) - np.median(good_old[:, 1])
                    mouse_worker.move_mouse(x=movement_x * 1.3, y=movement_y * 1.3)
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    pass # Optical flow error
                    first_frame = True
            first_frame = True
            old_gray = frame_gray.copy()

        if mask_window:
            cv2.imshow('antirecoil', resized)
            
        if debug_window:
            height = int(cfg.detection_window_height * cfg.debug_window_scale_percent / 100)
            width = int(cfg.detection_window_width * cfg.debug_window_scale_percent / 100)
            dim = (width, height)
            cv2.resizeWindow('antirecoil', dim)
            resized = cv2.resize(frame, dim, cv2.INTER_NEAREST)
            cv2.imshow('mask', frame_with_mask)
        
        if debug_window or mask_window:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    frames = Capture()
    mouse_worker = MouseThread()
    
    feature_params = dict(maxCorners=200, qualityLevel=0.5, minDistance=10, blockSize=10)
    lk_params = dict(winSize=(50, 50), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03))

    debug_window = False
    mask_window = False
    init()