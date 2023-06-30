from tqdm import tqdm
import torch
import torch.nn.functional as F

def train(model,device,train_loader,optimizer,scheduler):
    model.train()
    correct=0
    processed=0
    pbar= tqdm(train_loader)
    for batch_idx,(data,target) in enumerate(pbar):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()

        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()

        pred = output.argmax(dim=1,keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(f'Loss={loss.item()} Accuracy={100*correct/processed:0.2f}')
        accuracy=100*correct/processed
        return accuracy,loss.item()

def test(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output,target,reduction='sum').item()
            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
    accuracy= correct/len(test_loader.dataset)*100
    return accuracy,test_loss

