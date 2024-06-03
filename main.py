import pickle
import random

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, LinearLR
from torch.utils.data import random_split, DataLoader
from data import MidiDataset, NoteData, NetworkData
from focal_loss import FocalLoss, focal_loss
from generation2 import generate_midi, generate_seed_from_int, get_transpose_seed, create_midi_track
from model import EmbConvLstmAttn, EmbConvLstmAttnTest, EmbConvLstmAttnSimple, EmbConvLstmAttnFixed, \
    EmbConvLstmAttnFixed2, EmbConvLstmAttnFixed3, EmbConvLstmAttnFixedInc, EmbConvLstmAttnFixedInc2, EmbConvLstmAttnInc, \
    EmbConvLstmAttnsmall, EmbConvLstmAttnAlt, EmbConvLstmAttnImpv, EmbConvLstmAttnImpv4, EmbConvLstmAttnImpv3, \
    EmbConvLstmAttnAlt2, EmbConvLstmAttnAlt3, Incept, EmbConvLstmAttnImpv2, EmbConvLstmAttnImpv22, EmbConvLstmNew, \
    PolyCNNLSTM, EmbConvLstmNew2, PolyphonicLSTM, MidiLSTM, PolyCNNLSM, PolyCNNLSM2, EmbConvLstPoly, \
    EmbConvLstPoly2, MidiLSTM2D, MidiLSTM2DA, EmbConvLstPoly3, EmbConvLstPoly4, EmbConvLstPoly5, EmbConvLstPoly6, \
    EmbConvLstPoly7, EmbConvLstPoly8, EmbConvLstPoly65, EmbConvLstPoly655, EmbConvLstPoly656, NewSingleModel, \
    NewSingleModel2, EmbConvLstm, NewSingleModel3, NewSingleModel4, EmbConvLstPoly10, EmbConvLstPoly11, \
    EmbConvLstPolyNew, SimpleLstm
from train2 import get_notes, prepare_sequences, train, get_notes_single, evaluate
from torch.cuda.amp import GradScaler


def save_model(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }, str(path))


def save_training_data(note_data, network_data, file_path):
    data = {
        "note_data": note_data,
        "network_data": network_data
    }

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def save_gen_data(note_data, file_path):
    data = {
        "note_data": note_data,
    }

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_notes(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    note_data = data["note_data"]
    network_data = data["network_data"]

    return note_data, network_data


def prepare_seq_from_notes(note_data, seq_len, skip_amt):
    return prepare_sequences(note_data, sequence_length=seq_len, skip_amount=skip_amt)


def main_train():
    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint = False
    load_note = True
    prep_notes = False
    model_name = 'classical-v9'
    notes_name = 'classical_single'
    model_dir = '/mnt/chia_raid/models'
    load_epoch = 100
    batch_size = 128
    gen_interval = 10
    seq_len = 512
    skip_amt = 2


    if load_note:
        if prep_notes:
            note_data, network_data = load_notes(f'{model_dir}/{notes_name}_training_note_data.pkl')
            network_data = prepare_sequences(note_data, sequence_length=seq_len, skip_amount=skip_amt)
            save_training_data(note_data, network_data, f'{model_dir}/{notes_name}_training_note_data.pkl')
        else:
            note_data, network_data = load_notes(f'{model_dir}/{notes_name}_training_note_data.pkl')
        # network_data = prepare_sequences(note_data, sequence_length=64)  # Save sequences for easy reproducibility
    # save_notes(note_data, network_data, f'{model_dir}/{notes_name}_note_data.pkl')

    else:

        note_data = get_notes_single('data/classical/archive/all/', 6)
        network_data = prepare_sequences(note_data, sequence_length=seq_len,
                                         skip_amount=skip_amt)  # Save sequences for easy reproducibility
        save_training_data(note_data, network_data, f'{model_dir}/{notes_name}_training_note_data.pkl')
        save_gen_data(note_data, f'{model_dir}/{notes_name}_note_data.pkl')

    # Create Dataset
    dataset = MidiDataset(network_data)
    # Split dataset into training and validation set
    # torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print('train length:', train_size)

    print("Data Loaded")

    alpha_values = [1.0] * 130  # Assuming you have 130 classes
    alpha_values[0] = 0.3  # Lower weight for class 0 (padding)
    alpha_values[128] = 0  # Lower weight for class 0 (padding)
    alpha_values[129] = 0  # Lower weight for class 0 (padding)

    # Initialize the model
    model = SimpleLstm(note_data, dropout=0.7, bidirectional=False)
    model = model.to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), amsgrad=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    # Define the scheduler parameters
    T_0 = 40  # Number of epochs before the first restart
    T_mult = 1  # Multiplier for the number of epochs after each restart
    eta_min = 0.005  # Minimum learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min=eta_min)
    #scheduler = OneCycleLR(optimizer, base_momentum=0.85, max_momentum=0.95, max_lr=0.1, pct_start=0.2,
     #                      steps_per_epoch=len(train_dataloader), epochs=300)
   # scheduler = LinearLR(optimizer, start_factor=1, end_factor=1, last_epoch=1000)

    if load_checkpoint:
        state_dict = torch.load(f'{model_dir}/{model_name}_checkpoint_epoch_{load_epoch}.pth')
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        #scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    #
    # for param_group in optimizer.param_groups:
    #      param_group['lr'] = 1e-4
    # Intervals to save hte model and generate test outputs

    save_interval = 20  # Save every 50 epochs
    generate_interval = 2  # Generate MIDI every 10 epochs

    # Training loop
    epochs = 20000
    start_epoch = load_epoch if load_checkpoint else 0
    if load_checkpoint:
        start_epoch = state_dict['epoch'] + 1

    # if scheduler is not None:
    #     scheduler.step(start_epoch)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')
    print("Training...")
    for epoch in range(start_epoch, epochs):
        # Save checkpoint
        if epoch == 0 or (epoch + 1) % save_interval == 0:
            save_model(model, optimizer, scheduler, epoch, f'{model_dir}/{model_name}_checkpoint_epoch_{epoch + 1}.pth')

        t_loss, t_acc = train(model, train_dataloader, criterion, optimizer, device, note_data, scaler, batch_size,
                              scheduler=scheduler, clip_value=5)
        v_loss, v_acc = evaluate(model, val_dataloader, criterion, device, note_data, batch_size)
        with open(f"models/{model_name}_training.txt", "a") as file:
            file.write(f"Epoch {epoch}: T_Lose: {t_loss}\tT_Acc: {t_acc} \t\tV_Lose: {v_loss}\tV_Acc: {v_acc}\n")

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f"Train Loss:{t_loss:.4f} Train Acc:{t_acc}\t|\tValid Loss:{v_loss:.4f} Valid Acc:{v_acc}")

        # Generate test midi
        if (epoch + 1) % gen_interval == 0:
            generate_midi(model, note_data, network_data, f'midi/{model_name}_training_gen_{epoch + 1}.mid',
                          temperature=0.7, seq_len=600)


def main_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model_name = 'video-game-4-4-poly3-fix'
    notes_name = 'large_single'
    model_dir = '/mnt/buffer/models'
    load_epoch = 220

    note_data, network_data = load_notes(f'{model_dir}/{notes_name}_note_data.pkl')
    model = EmbConvLstPoly5(note_data)
    model = model.to(device)
    state_dict = torch.load(f'{model_dir}/{model_name}_checkpoint_epoch_{load_epoch}.pth')
    model.load_state_dict(state_dict['model_state_dict'])

    keys = ['A', 'A#', 'B', 'B#', 'C', 'C#', 'E', 'E#', 'F', 'F#', 'G', 'G#']

    i = 0
    for input in network_data.input:
        pass
        # create_midi_track(generate_notes_test(note_data, input), output_file=f'{i}.mid', )
        # i +=1

    # for i in range(0, 100):
    #     print(f'Generating #{i}')
    #     # seed = generate_seed_from_int(random.randint(0, 65000), 64, note_data)
    #     # seed = seed.to(device)
    #    # key = keys[random.randint(0, len(keys))]
    #     #minor = random.choice([True, False])
    #     #rnd_seed = get_transpose_seed(note_data,64, key, device, minor=minor)
    #     generate_midi(model, note_data, network_data, f'midi/generations/poly_{model_name}_{i}.mid',
    #                   seed=None, temperature=1, seq_len=200)
    #


def review_notes():
    notes_name = 'single_notes_note_data'
    note_data, network_data = load_notes(f'models/{notes_name}.pkl')
    tn = note_data.training_notes
    print(tn)


def gen_network_data():
    note_data = get_notes('/home/mindspice/code/Python/PycharmProjects/midi_generator/data/classical/archive/beeth/',
                          get_flat=True)
    network_data = prepare_sequences(note_data)  # Save sequences for easy reproducibility
    save_training_data(note_data, network_data, f'models/{"beeth"}_note_data.pkl')


# import torch.nn.functional as F
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
#
#         if self.reduction == 'sum':
#             return F_loss.sum()
#         elif self.reduction == 'mean':
#             return F_loss.mean()

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-CE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss
if __name__ == "__main__":
    main_train()
