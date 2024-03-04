#definisanje modela koji ce da se koriste za klasifikaciju
#koristi se transfer learning, pri cemu se vrsi fine-tjunovanje nad datim skupom podataka
#https://pytorch.org/tutorials/recipes/quantization.html

import torch
import torchvision
from torchvision.models import resnet50
import joblib
from utils import check_cuda
import os
   
def model_resnet_trans_learn(quantization: str):
  device = check_cuda()
  if quantization == 'yes_quant':
    model = torchvision.models.quantization.resnet50(pretrained=True, quantize = True)
  else:
    model = resnet50(pretrained=True).to(device)

  #proces transfer-learninga kroz zamrzavanje i odmrzavanje odgovarajucih slojeva
  for param in model.parameters():
      param.requires_grad = False
  for param in model.layer4.parameters():
      param.requires_grad = True
  model.classifier = torch.nn.Sequential(
      torch.nn.Dropout(p=0.3, inplace=True),
      torch.nn.Linear(in_features=25088,
                      out_features=8, # broj izlaza jednak broju klasa
                      bias=True)).to(device)
  return model

#ucitavanje modela
def load_model(file_path:str):
   return joblib.load(file_path)
#pamcenje modela
def save_model(model, file_path:str):
   joblib.dump(model, file_path)
#provera velicine modela
def print_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')