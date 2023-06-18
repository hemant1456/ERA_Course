# importing modules
import torch
import torch.nn.functional as F


def train(model,device,train_loader,optimizer,epoch):
    model.train()
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # move the data to device
        data, target = data.to(device), target.to(device)
        # zero the gradients
        optimizer.zero_grad()
        # forward pass
        output = model(data)
        # calculate the loss
        loss = F.nll_loss(output, target)
        # backward pass
        loss.backward()
        # update the weights
        optimizer.step()
        # print the progress
        correct += output.argmax(dim=1).eq(target).sum().item()
        processed += len(data)
        if batch_idx % 200 == 0:
            print(f'Epoch: {epoch+1} and batch idx: {batch_idx} Train loss is {loss.item()} Train Accuracy: {100*correct/processed:0.2f}')

def test(model,device,test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output,target,reduction='sum').item()
            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'Test loss is {test_loss} and Test accuracy is {100*correct/len(test_loader.dataset)}')

def run(model,device,train_loader,test_loader,optimizer,epochs,scheduler):
    for epoch in range(epochs):
        print(f'\ncurrently we are at epoch {epoch+1} and learning rate used is {scheduler.get_last_lr()[0]}')
        train(model,device,train_loader,optimizer,epoch)
        test(model,device,test_loader)
        scheduler.step()