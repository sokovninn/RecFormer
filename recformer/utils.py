import torch
from torch import nn
device = torch.device('cuda')


def predict_on_batch(model, batch, multitask_switch=None):
  x = batch[0].to(device=device)
  if multitask_switch == "cuisine":
    prediction, _ = model(x)
  elif multitask_switch == "ingredients":
    _, prediction = model(x)
  else:
    prediction = model(x)
  probs = nn.functional.softmax(prediction, dim=1)
  return probs

def measure_accuracy(model, data_loader, multitask_switch=None):
  correct_preds = 0
  model.eval()
  with torch.no_grad():
    for batch in data_loader:
        preds = predict_on_batch(model, batch, multitask_switch)
        target = batch[1].to(device=device).flatten()
        correct_preds += torch.sum(torch.argmax(preds, axis=1) == target)
    accuracy = correct_preds / len(data_loader.dataset)
  return accuracy