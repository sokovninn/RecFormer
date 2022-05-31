import torch
from torch import nn
import tqdm

def train(model, criterion, optimizer, data_loader, validation_loader, epochs, clipping_on=False, device='cuda'):
  model.to(device=device)
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm.tqdm(data_loader):
      x = batch[0].to(device=device)
      #print(x[0])
      y = batch[1].flatten().to(device=device)
      preds = model(x)
      #print(preds.shape, y.shape)
      loss = criterion(preds, y)
      optimizer.zero_grad()
      loss.backward()
      if clipping_on:
        nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()
      running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)

    val_loss = 0.0
    model.eval()
    with torch.no_grad():
      for batch in validation_loader:
          
          x = batch[0].to(device=device)
          y = batch[1].flatten().to(device=device)
          preds = model(x)
          loss = criterion(preds, y)
          val_loss += loss.item()
      val_loss /= len(validation_loader)
        
    print('Epoch: {}, Training Loss: {}, Validation Loss: {}'.format(epoch, epoch_loss, val_loss))
    #print(measure_accuracy(model, train_loader))
    accuracy = measure_accuracy(model, validation_loader)
    print(accuracy)

def train_multitask(model, criterion, optimizer, data_loader, validation_loader_cuisine, validation_loader_ingredients, epochs, clipping_on=False, device='cuda', loss_weights=[1,1]):
  model.to(device=device)
  best_cls_acc = 0
  best_cmp_acc = 0
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch in  tqdm.tqdm(data_loader):
      x = batch[0].to(device=device)
      #print(x[0])
      #print(batch[1].shape)
      y_cuisine = batch[1][:,:,0].flatten().to(device=device)
      y_ingredients = batch[1][:,:,1].flatten().to(device=device)
      preds_cuisine, preds_ingredients = model(x)
      #print(preds.shape, y.shape)
      loss = loss_weights[0] * criterion(preds_cuisine, y_cuisine) + loss_weights[1] * criterion(preds_ingredients, y_ingredients)
      optimizer.zero_grad()
      loss.backward()
      if clipping_on:
        nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()
      running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)

    val_loss_cuisine = 0.0
    val_loss_ingredients = 0.0
    model.eval()
    with torch.no_grad():
      for batch in validation_loader_cuisine:
          x = batch[0].to(device=device)
          y = batch[1].flatten().to(device=device)
          preds_cuisine, preds_ingredients = model(x)
          loss = criterion(preds_cuisine, y)
          val_loss_cuisine += loss.item()
            
      for batch in validation_loader_ingredients:
          x = batch[0].to(device=device)
          y = batch[1].flatten().to(device=device)
          preds_cuisine, preds_ingredients = model(x)
          loss = criterion(preds_ingredients, y)
          val_loss_ingredients += loss.item()
      val_loss_cuisine /= len(validation_loader_cuisine)
      val_loss_ingredients /= len(validation_loader_ingredients)
        
    print('Epoch: {}, Training Loss: {}, Validation Loss Cuisine: {}, Validation Loss Ingredients: {}'.format(epoch, epoch_loss, val_loss_cuisine, val_loss_ingredients))
    #print(measure_accuracy(model, train_loader))
    cls_acc = measure_accuracy(model, validation_loader_cuisine, multitask_switch="cuisine")
    cmp_acc = measure_accuracy(model, validation_loader_ingredients, multitask_switch="ingredients")
    print("Validation classification accuracy: {}".format(cls_acc))
    print("Validation completion accuracy: {}".format(cmp_acc))
    if cls_acc >= best_cls_acc and cmp_acc >= best_cmp_acc:
        torch.save(model.state_dict(), "RecFormer_multitask_best.pth")
        best_cls_acc = cls_acc
        best_cmp_acc = cmp_acc
        print("Best model saved")