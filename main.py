import pickle
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import random_split, DataLoader
from data import MidiDataset
from generation import generate_midi
from model import EmbConvLstmAttn
from train import get_notes, prepare_sequences, train, validate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint = False
    load_note = True
    model_name = 'video_game-4-4-attn-2E'
    notes_name = 'large_note_data'
    load_epoch = 100

    if load_checkpoint:
        state_dict = torch.load(f'models/{model_name}_checkpoint_epoch_{load_epoch}.pth')

    if load_note:
        note_data, network_data = load_notes(f'models/{notes_name}.pkl')
    else:
        note_data = get_notes('data/time_sig/4-4/')
        network_data = prepare_sequences(note_data)  # Save sequences for easy reproducibility
        save_notes(note_data, network_data, f'models/{model_name}_note_data.pkl')

    # Create Dataset
    dataset = MidiDataset(network_data)
    print("vocab_size", note_data.d_vocab)
    # Split dataset into training and validation set
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    print('train length:', train_size)

    # Initialize the model
    model = EmbConvLstmAttn(note_data, bidirectional=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Define the scheduler parameters
    T_0 = 10  # Number of epochs before the first restart
    T_mult = 2  # Multiplier for the number of epochs after each restart
    eta_min = 1e-6  # Minimum learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)


    if load_checkpoint:
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    # Intervals to save hte model and generate test outputs
    save_interval = 25  # Save every 50 epochs
    generate_interval = 5  # Generate MIDI every 10 epochs

    # Training loop
    epochs = 20000
    start_epoch = 0
    if load_checkpoint:
        start_epoch = state_dict['epoch'] + 1

    if scheduler is not None:
        scheduler.step(start_epoch)

    print("Training...")
    for epoch in range(start_epoch, epochs):
        # Save checkpoint
        if epoch == 0 or (epoch + 1) % save_interval == 0:
            save_model(model, optimizer, scheduler, epoch, f"models/{model_name}_checkpoint_epoch_{epoch + 1}.pth")

        t_loss, t_acc = train(model, train_dataloader, criterion, optimizer, scheduler, device, note_data)
        v_loss, v_acc = validate(model, val_dataloader, criterion, device, note_data)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f"Train Loss:{t_loss:.4f} Train Acc:{t_acc}\t|\tValid Loss:{v_loss:.4f} Valid Acc:{v_acc}")

        # Generate test midi
        if (epoch + 1) % generate_interval == 0:
            generate_midi(model, note_data, network_data, f'midi/{model_name}_training_gen_{epoch + 1}.mid',
                          temperature=0.7)


def save_model(model, optimizer, scheduler, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
    }, str(path))


def save_notes(note_data, network_data, file_path):
    data = {
        "note_data": note_data,
        "network_data": network_data
    }

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_notes(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    note_data = data["note_data"]
    network_data = data["network_data"]

    return note_data, network_data


if __name__ == "__main__":
    main()