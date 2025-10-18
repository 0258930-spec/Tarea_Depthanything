import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

cap = cv2.VideoCapture(0)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

if DEVICE == 'cuda':
    print(f"[INFO] GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Versión CUDA de PyTorch: {torch.version.cuda}")
else:
    print("[WARN] No se detectó CUDA, usando CPU.")
    
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load("Pesos/depth_anything_v2_vits.pth"))
model = model.to(DEVICE).eval()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    depth = model.infer_image(frame)  # HxW raw depth map in numpy
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    combined = np.hstack((frame, depth_colored))
    cv2.imshow('Depth Estimation', combined)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()