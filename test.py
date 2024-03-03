import torch
import numpy as np

model = torch.load('model/my_mnist_model.pt')
model.eval()

from PIL import Image
import torchvision.transforms as transforms

image_path = 'test/7_mnist.jpg'

image = Image.open(image_path).convert('L')
# Flatten the image  
image = image.resize((28,28))
image = np.array(image) 

# Reshape to add channel dim
image = image.reshape(1, 28, 28)

transform_obj = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = transform_obj(image).unsqueeze(0)

# Flatten image to match model input shape
image = image.reshape(image.shape[0], -1) 

with torch.no_grad():
    logps = model(image) 

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
predicted_digit = probab.index(max(probab))

print("Predicted Digit =", predicted_digit)