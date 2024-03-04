#grad-cam importi
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils import denormalize_image

def get_grad_cam(img, model):
    #ovde je target layer poslednji konvolucioni sloj
    target_layers = [model.layer4[-1]]

    input_tensor = img.unsqueeze(0)
    cam = GradCAM(model = model, target_layers = target_layers)
    targets = None #na ovaj nacin ce da se koristi kategorija sa najvecim score-om
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    denormalized_img = denormalize_image(img)
    final_grad_cam = show_cam_on_image(denormalize_image, grayscale_cam, use_rgb=True)
    return final_grad_cam
