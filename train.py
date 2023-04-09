from torch import nn
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path
from torch.optim.lr_scheduler import ExponentialLR

from src.mel_dataset import get_dataloader
from src.model import UMelResNet
from src.utils import TrainingLogging, save_images
from src.settings import *


def train_unet(net, optimizer, criterion_cl, criterion_dn, train_loader, val_loader,
               scheduler=None, n_epochs=41, display_step=5):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_accuracy = 0
    logs = TrainingLogging()
    best_model_path = None
    model_weights_path = './logs/weights'
    Path(model_weights_path).mkdir(parents=True, exist_ok=True)
    for epoch in range(n_epochs):

        # train one epoch
        epoch_loss = 0.0
        num_items = 0
        for sample_clean, sample_noisy in tqdm(train_loader):
            labels = [0] * len(sample_clean)
            labels.extend([1] * len(sample_noisy))
            labels = torch.tensor(labels).to(DEVICE)
            data_input = torch.cat([sample_clean, sample_noisy], dim=0).to(DEVICE)
            data_output = torch.cat([sample_clean, sample_clean], dim=0).to(DEVICE)
            current_num_items = len(labels)
            num_items += current_num_items
            data_input = data_input.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            x_classification, x_mel = net(data_input)
            loss_cl = criterion_cl(x_classification, labels)
            loss_dn = criterion_dn(x_mel, data_output)
            loss = loss_cl + loss_dn
            epoch_loss += (loss.item() * current_num_items)
            loss.backward()
            optimizer.step()
        avg_loss = epoch_loss / num_items
        print(f'epoch - {epoch}, train_loss - {avg_loss}')

        if scheduler is not None and (epoch / n_epochs) > 0.5:
            scheduler.step()

        # validation
        if epoch % display_step == 0:
            epoch_loss = 0.0
            epoch_loss_dn = 0.0
            epoch_accuracy = 0.0
            num_items = 0
            first = True
            for sample_clean, sample_noisy in val_loader:
                labels = [0] * len(sample_clean)
                labels.extend([1] * len(sample_noisy))
                labels = torch.tensor(labels).to(DEVICE)
                data_input = torch.cat([sample_clean, sample_noisy], dim=0).to(DEVICE)
                data_output = torch.cat([sample_clean, sample_clean], dim=0).to(DEVICE)
                current_num_items = len(labels)
                num_items += current_num_items
                with torch.no_grad():
                    x_classification, x_mel = net(data_input)
                    loss_cl = criterion_cl(x_classification, labels)
                    loss_dn = criterion_dn(x_mel, data_output)
                    loss = loss_cl + loss_dn

                epoch_loss += (loss.item() * current_num_items)
                epoch_loss_dn += (loss_dn.item() * current_num_items)
                epoch_accuracy += ((sum(
                    torch.argmax(x_classification, dim=1) == labels) / len(labels)).item() * current_num_items)

                if first:
                    images = [data_input[0],
                              x_mel[0],
                              data_input[BATCH_SIZE],
                              x_mel[BATCH_SIZE]]
                    save_images(images, path=f'./logs/mel_epoch-{epoch}.png')
                    first = False

            avg_vloss = epoch_loss / num_items
            avg_vloss_dn = epoch_loss_dn / num_items
            avg_accuracy = epoch_accuracy / num_items
            print(f'accuracy - {avg_accuracy}, val_loss - {avg_vloss}, val_loss_dn - {avg_vloss_dn}')
            logs.add('accuracy', avg_accuracy, epoch)
            logs.add('val_loss', avg_vloss, epoch)
            logs.add('train_loss', avg_loss, epoch)
            logs.add('val_dn_loss', avg_vloss_dn, epoch)

            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_model_path = f'{model_weights_path}/model_{timestamp}_{epoch}_{round(best_accuracy, 5)}'
                print(f'current_best_model_path - {best_model_path}')
                torch.save(net.state_dict(), best_model_path)

    return logs, best_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')
    required.add_argument("-t", "--train_folder", help="train folder with clean and noisy directories",
                          type=str, required=True)

    required.add_argument("-v", "--validation_folder", help="validation folder with clean and noisy directories",
                          type=str, required=True)

    args = parser.parse_args()
    train_path = Path(args.train_folder)
    val_path = Path(args.validation_folder)
    print(f'Your device is {DEVICE}')
    print('Dataloader creation ...')
    train_loader, val_loader = get_dataloader(train_path, val_path)
    print('Creation completed')

    criterion_cl = nn.CrossEntropyLoss()
    criterion_dn = nn.MSELoss()
    net = UMelResNet(HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    print('Training started ...')
    logs, best_model_path = train_unet(net, optimizer, criterion_cl, criterion_dn, train_loader, val_loader,
                                       scheduler, n_epochs=25, display_step=2)
    print('Training finished!')
    logs.plot('val_loss', 'train_loss', 'val_dn_loss', path='./logs/loss.png')
    logs.plot('accuracy', path='./logs/accuracy.png')
    print(f'The best model path - {best_model_path}')
