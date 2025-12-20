import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

def find_edge_channel(img):
    edges_mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    width = img.shape[1]
    height = img.shape[0]
    
    gray_im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    med1 = np.median(gray_im[:height//2,:width//2])
    med2 = np.median(gray_im[:height//2,width//2:])
    med3 = np.median(gray_im[height//2:,width//2:])
    med4 = np.median(gray_im[height//2:,:width//2])

    l1 = int(max(0,(1-0.205)*med1))
    u1 = int(min(255,(1+0.205)*med1))
    e1 = cv2.Canny(gray_im[:height//2,:width//2],l1,u1)

    l2 = int(max(0,(1-0.205)*med2))
    u2 = int(min(255,(1+0.205)*med2))
    e2 = cv2.Canny(gray_im[:height//2,width//2:],l2,u2)

    l3 = int(max(0,(1-0.205)*med3))
    u3 = int(min(255,(1+0.205)*med3))
    e3 = cv2.Canny(gray_im[height//2:,width//2:],l3,u3)

    l4 = int(max(0,(1-0.205)*med4))
    u4 = int(min(255,(1+0.205)*med4))
    e4 = cv2.Canny(gray_im[height//2:,:width//2],l4,u4)

    edges_mask[:height//2,:width//2] = e1
    edges_mask[:height//2,width//2:] = e2
    edges_mask[height//2:,width//2:] = e3
    edges_mask[height//2:,:width//2] = e4
    
    edges_mask_inv = cv2.bitwise_not(edges_mask)
    
    return edges_mask, edges_mask_inv

def load_model(model_path):
    model = UNet()
    save_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(save_dict["model"])
    model = model.to(device)
    model.eval()
    return model, save_dict

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((180, 330)),
        transforms.ToTensor()
    ])

    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges, edges_inv = find_edge_channel(image)
    
    output_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    output_image[:, :, 0] = gray
    output_image[:, :, 1] = edges
    output_image[:, :, 2] = edges_inv
    
    processed = transform(output_image)
    return processed, original_image

def evaluate(model, image):
    processed_img, original_img = preprocess_image(image)
    
    input_tensor = processed_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output)
        mask_pred = (pred > 0.5).float()

    mask_np = mask_pred.squeeze().cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    
    return mask_np


if __name__ == "__main__":
    # moved segmentation visualizations here to test the evaluate file when run directly, otherwise this file is used for perception.py
    import os
    import glob
    from Perception.model.unet import UNet
    
    model_path = "Perception/model/epoch_39.pt"
    test_images_dir = "data/custom/inputs"
    
    def load_model(model_path):
        model = UNet()
        save_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(save_dict["model"])
        model = model.to(device)
        model.eval()
        return model
    
    def main():
        model = load_model(model_path)
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        test_images = []
        for ext in image_extensions:
            test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
        
        for image_path in test_images:
            img_name = os.path.basename(image_path)
            print(f"\nEvaluating: {img_name}")
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                
                pred_mask = evaluate(model, image)
                print(f"Prediction shape: {pred_mask.shape}")
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    
    main()
