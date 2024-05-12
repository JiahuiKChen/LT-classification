from data.coco import COCODataset
from data.pascal import PASCALDataset
# from semantic_aug.datasets.caltech101 import CalTech101Dataset
from data.flowers102 import Flowers102Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from itertools import product
from tqdm import trange
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as distributed
import wandb

import argparse
import pandas as pd
import numpy as np
import random
import os


DATASETS = {
    "coco": COCODataset, 
    "pascal": PASCALDataset,
    # "caltech": CalTech101Dataset,
    "flowers": Flowers102Dataset
}

def run_experiment(cond_method: str = "embed_cutmix_dropout",
                   examples_per_class: int = 0, 
                   seed: int = 0, 
                   dataset: str = "coco", 
                   iterations_per_epoch: int = 200, 
                   num_epochs: int = 50, 
                   batch_size: int = 32, 
                   synthetic_probability: float = 0.5, 
                   classifier_backbone: str = "resnet50"):

    run = wandb.init(
        mode="disabled",
        project="SynthFineTune",
        config={
             "cond_method": cond_method,
             "dataset": dataset,
             "examples_per_class": examples_per_class,
             "trial": seed,
             "num_epochs": num_epochs,
             "batch_size": batch_size,
             "classifier_backbone": classifier_backbone,
             "synthetic_probability": synthetic_probability
        },
        name=f"{dataset}{examples_per_class}ex_{cond_method}_trial{seed}",
        group=f"{dataset}_{cond_method}"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = DATASETS[dataset](
        cond_method=cond_method,
        split="train", examples_per_class=examples_per_class, 
        synthetic_probability=synthetic_probability)

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler, num_workers=4)

    val_dataset = DATASETS[dataset](
        cond_method=cond_method,
        split="val", seed=seed)

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, 
        sampler=val_sampler, num_workers=4)

    model = ClassificationModel(
        train_dataset.num_classes, 
        backbone=classifier_backbone
    )
    model = nn.DataParallel(model).cuda()

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    records = []
    best_val_acc = 0.0

    for epoch in trange(num_epochs, desc="Training Classifier"):

        model.train()

        epoch_loss = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')

        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            if len(label.shape) > 1: label = label.argmax(dim=1)

            accuracy = (prediction == label).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            with torch.no_grad():
            
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        training_loss = epoch_loss / epoch_size.clamp(min=1)
        training_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        training_loss = training_loss.cpu().numpy()
        training_accuracy = training_accuracy.cpu().numpy()

        # VAL
        model.eval()

        epoch_loss = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes, 
            dtype=torch.float32, device='cuda')

        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            accuracy = (prediction == label).float()

            with torch.no_grad():
            
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_loss = epoch_loss / epoch_size.clamp(min=1)
        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        validation_loss = validation_loss.cpu().numpy()
        validation_accuracy = validation_accuracy.cpu().numpy()

        epoch_train_loss = training_loss.mean() 
        epoch_train_acc = training_accuracy.mean() 
        epoch_val_loss = validation_loss.mean()
        epoch_val_acc = validation_accuracy.mean()

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

        print(f"Epoch {epoch}\tTrain Acc: {epoch_train_acc}\tTrain Loss: {epoch_train_loss}\tVal Acc: {epoch_val_acc}\tVal Loss: {epoch_val_loss}")
        run.log(
            {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc 
            }
        )

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=epoch_train_loss, 
            metric="Loss", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=epoch_val_loss, 
            metric="Loss", 
            split="Validation"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=epoch_train_acc, 
            metric="Accuracy", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=epoch_val_acc, 
            metric="Accuracy", 
            split="Validation"
        ))

        for i, name in enumerate(train_dataset.class_names):

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_loss[i], 
                metric=f"Loss {name.title()}", 
                split="Training"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_loss[i], 
                metric=f"Loss {name.title()}", 
                split="Validation"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_accuracy[i], 
                metric=f"Accuracy {name.title()}", 
                split="Training"
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_accuracy[i], 
                metric=f"Accuracy {name.title()}", 
                split="Validation"
            ))

    print(f"\nTrial {seed} with examples_per_class={examples_per_class} finished {num_epochs} epochs fine-tuning")
    print(f"\tBest Val Accuracy: {best_val_acc}") 
    run.finish()

    return records


class ClassificationModel(nn.Module):
    
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        
        super(ClassificationModel, self).__init__()

        self.backbone = backbone
        self.image_processor  = None
        
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.out = nn.Linear(2048, num_classes)
        
    def forward(self, image):
        
        x = image

        with torch.no_grad():

            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            
        return self.out(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Fine Tuning")

    parser.add_argument("--cond-method", type=str, default="embed_cutmix_dropout", 
                        choices=["rand_img_cond", "cutmix", "cutmix_dropout", "dropout", "embed_cutmix",
                                 "embed_cutmix_dropout", "embed_mixup", "embed_mixup_dropout",
                                 "mixup", "mixup_dropout"])
    
    parser.add_argument("--synthetic-probability", type=float, default=0.5)
    # parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--classifier-backbone", type=str, 
                        default="resnet50", choices=["resnet50"])

    parser.add_argument("--iterations-per-epoch", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    
    parser.add_argument("--dataset", type=str, default="coco", 
                        choices=["coco", "pascal", "flowers", "caltech"])

    args = parser.parse_args()

    log_dir = os.path.join(f"./logs/{args.dataset}", args.cond_method)
    os.makedirs(log_dir, exist_ok=True)

    all_trials = []

    options = product(range(args.num_trials), args.examples_per_class)
    options = np.array(list(options))
    # options = np.array_split(options, world_size)[rank]

    for trial, examples_per_class in options.tolist():

        hyperparameters = dict(
            cond_method=args.cond_method,
            examples_per_class=examples_per_class,
            seed=trial, 
            dataset=args.dataset,
            num_epochs=args.num_epochs,
            iterations_per_epoch=args.iterations_per_epoch, 
            batch_size=args.batch_size,
            synthetic_probability=args.synthetic_probability, 
            classifier_backbone=args.classifier_backbone)

        all_trials.extend(run_experiment(**hyperparameters))

        path = f"results_{trial}_{examples_per_class}.csv"
        path = os.path.join(log_dir, path)

        pd.DataFrame.from_records(all_trials).to_csv(path)
        print(f"[conditioning method= {args.cond_method}  n={examples_per_class}  saved to:  {path}")