from models import UNET , transform
import matplotlib.pyplot as plt
import torch 
from PIL import Image
device = torch.device('cpu')


model = UNET().to(device)
model.load_state_dict(torch.load(r"/run/media/bhavesh/Shared/backup/pathpal/road_navigation/segmentation_torch_unet.pth",map_location=torch.device('cpu'),weights_only=True))
img_path = r"/run/media/bhavesh/Shared/backup/pathpal/road_navigation/roadtest1.jpg"

model.eval()

img = Image.open(img_path)
final_img = transform(img)

final_img= final_img.unsqueeze(0).to(device)

mask = model(final_img)

output = mask > 0.5  
output = output.squeeze(0)
output = output.permute(1,2,0)

plt.imshow(output)
plt.show()