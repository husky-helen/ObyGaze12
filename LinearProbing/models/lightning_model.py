import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import io



class DynMLP2HiddenLit(pl.LightningModule):

    def __init__(self,layers_config, activation = nn.ReLU(),w = None, average = 'weighted'):
        """MLP model

        Args:
            layers_config (list): _description_
            activation (torch function, optional): Activation function to use. Defaults to nn.ReLU().
            w (torch.tensor, optional): weights to use in the loss. Defaults to None.
            average (str) : average to use in the computation of the metrics (f1 score, recall, precision)
        """
        super(DynMLP2HiddenLit, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(len(layers_config)-1):
            self.layers.append(nn.Linear(layers_config[i], layers_config[i+1]))

            if i < len(layers_config) - 2:
                self.layers.append(self.activation)

        self.loss_ce = nn.CrossEntropyLoss(weight = w)
        self.average = average


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
   
        return torch.softmax(x,dim=1)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_ce(outputs, targets)
        self.log("train_loss",loss,reduce_fx = "sum",on_step=True,on_epoch=True)
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_ce(outputs, targets)
        
        acc = accuracy(outputs, targets,task='multiclass', num_classes = targets.shape[1])
        pred_label = torch.argmax(outputs, dim=1)
        target_classes = torch.argmax(targets, dim=1)
        acc_ = accuracy_score(target_classes.cpu(), pred_label.cpu())
        f1 = f1_score(target_classes.cpu(), pred_label.cpu(), average=self.average)
        recall = recall_score(target_classes.cpu(), pred_label.cpu(), average=self.average)
        precision = precision_score(target_classes.cpu(), pred_label.cpu(), average=self.average, zero_division=0)
        
        self.log_conf_matrix(target_classes, pred_label, "Val Confusion Matrix")
        self.log('val_loss',loss)
        self.log("val_acc",acc)
        self.log("val_acc_2", acc_, on_step=True, on_epoch=True)
        self.log('val_f1', f1)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        
        return loss
    
    def log_conf_matrix(self, target_classes, pred_label, title='Confusion Matrix'):
        """Function to be able to load a plot of the confusion matrix in tensorboard

        Args:
            target_classes (torch.tensor): Ground truth classes
            pred_label (torch.tensor): Predicted classes
            title (str, optional): Title of the plot. Defaults to 'Confusion Matrix'.
        """
        confusion = confusion_matrix(target_classes.cpu(), pred_label.cpu())
        fig = plt.figure()
        sn.heatmap(confusion, annot=True, fmt='g')  
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')

        # Save the figure as an image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0) 
        pil_img = Image.open(buf)

        transform = transforms.ToTensor()
        tensor_img = transform(pil_img)


        #Load image in tensorboard
        self.logger.experiment.add_image(title, tensor_img, global_step=self.global_step)

        buf.close()
        plt.close()



    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_ce(outputs, targets)
    
        pred_label = torch.argmax(outputs, dim=1)
        target_classes = torch.argmax(targets, dim=1)
        acc_ = accuracy_score(target_classes.cpu(), pred_label.cpu())
        f1 = f1_score(target_classes.cpu(), pred_label.cpu(), average=self.average)
        recall = recall_score(target_classes.cpu(), pred_label.cpu(), average=self.average)
        precision = precision_score(target_classes.cpu(), pred_label.cpu(), average=self.average,zero_division=0)
        
        self.log_conf_matrix(target_classes, pred_label, "Test Confusion Matrix")
        self.log("test_acc_2", acc_)
        self.log('test_f1', f1)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
        
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)