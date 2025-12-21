import torch
import cv2
import numpy as np
from torchvision import transforms
from Perception.model.unet import UNet

class UNetEvaluator:
    def __init__(self, weights_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet()
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

        # Notice: No ToPILImage needed if we handle the resize manually to keep control
        self.transform = transforms.ToTensor()

    def find_edge_channel(self, img):
        height, width = img.shape[:2]
        gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges_mask = np.zeros((height, width), dtype=np.uint8)

        # Separate into quadrants to calculate adaptive Canny thresholds
        # This must match your dataloader exactly
        quads = [
            (0, height//2, 0, width//2),    # Top-Left
            (0, height//2, width//2, width), # Top-Right
            (height//2, height, width//2, width), # Bottom-Right
            (height//2, height, 0, width//2)  # Bottom-Left
        ]

        for (y1, y2, x1, x2) in quads:
            sub = gray_im[y1:y2, x1:x2]
            med = np.median(sub)
            lower = int(max(0, (1 - 0.205) * med))
            upper = int(min(255, (1 + 0.205) * med))
            edges_mask[y1:y2, x1:x2] = cv2.Canny(sub, lower, upper)

        edges_mask_inv = cv2.bitwise_not(edges_mask)
        return edges_mask, edges_mask_inv

    def evaluate(self, cv_image):
        # 1. Build the 3-channel feature map expected by the model
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges, edges_inv = self.find_edge_channel(cv_image)
        
        # Stack them into (H, W, 3)
        input_stack = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        input_stack[:, :, 0] = gray
        input_stack[:, :, 1] = edges
        input_stack[:, :, 2] = edges_inv

        # 2. Resize to the training resolution (180, 330)
        # CV2 uses (width, height)
        input_resized = cv2.resize(input_stack, (330, 180), interpolation=cv2.INTER_AREA)

        # 3. Transform to Tensor (0.0 - 1.0)
        input_tensor = self.transform(input_resized).unsqueeze(0).to(self.device)

        # 4. Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.sigmoid(output)

        output_np = pred.squeeze().cpu().numpy()
        bw_mask = (output_np > 0.5).astype(np.uint8) * 255
        
        return bw_mask