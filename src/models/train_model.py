import time
import json
import math
import mlflow
from numpy import dtype, hypot
import pandas as pd
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from torch import embedding, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.data.dataloader import ImageCaptionDataset
from src.utils import (save_checkpoint,
                       adjust_learning_rate,
                       AverageMeter,
                       clip_gradient,
                       get_text_for_batch,
                       log_scalar,
                       BaseLogger)
from nltk.translate.bleu_score import corpus_bleu
from src.models.models import EncoderCNN, DecoderRNN
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import Sequential, AddToken, Truncate, ToTensor, VocabTransform
from nltk.translate.bleu_score import corpus_bleu


# Data parameters
data_path = Path(__file__).resolve().parents[2].joinpath("data", "raw")
caption_prediction_path = data_path.joinpath("caption-prediction")

# Model parameters
embedding_size = 512  # dimension of word embeddings
hidden_size = 512
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
decoder_layers = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 30  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 64
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-3  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
tokenizer = get_tokenizer('basic_english')

corpus = pd.read_csv(caption_prediction_path.joinpath("corpus.csv"), sep="\t").loc[:, "caption"].values


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


UNKNOWN_TOKEN = "<unk>"
START_TOKEN = "<sos>"
END_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
MAX_SEQ_LEN = 128

vocabulary = build_vocab_from_iterator(yield_tokens(corpus), specials=[UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, PAD_TOKEN])
vocabulary.set_default_index(vocabulary[UNKNOWN_TOKEN])
vocab_size = vocabulary.__len__()


def collate_function(batch):
    """
    create a mini batch an make sure the image are of the same size
    """
    batch.sort(key=lambda data: len(data[1]), reverse=True) # sort by the longest caption
    images, captions = zip(*batch) # unzip the batch
    images = torch.stack(images) # stack the images
    captions_lengths = [len(caption) for caption in captions]
    captions_array = np.full((len(captions), max(captions_lengths)), vocabulary.lookup_indices([PAD_TOKEN])[0])
    captions_tensor = torch.from_numpy(captions_array)
    for index, caption in enumerate(captions):
        end = captions_lengths[index]
        captions_tensor[index, :end] = caption[:end]
    
    return images.to(device), captions_tensor.to(device), captions_lengths


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    text_tranform = Sequential(VocabTransform(vocabulary),
                               Truncate(max_seq_len=MAX_SEQ_LEN - 2),
                               ToTensor())

    # Initialize / load checkpoint
    if checkpoint is None:
        encoder = EncoderCNN(embedding_size=embedding_size)
        decoder = DecoderRNN(embed_size=embedding_size,
                             hidden_size=hidden_size,
                             vocab_size=vocab_size,
                             num_layers=decoder_layers)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Custom dataloaders
    training_dataset = ImageCaptionDataset(caption_path=caption_prediction_path.joinpath("training-captions.csv"),
                                           images_dir=data_path.joinpath("training-images", "train"),
                                           sequence_length=200,
                                           text_tokenizer=tokenizer,
                                           text_transform=text_tranform)
    validation_dataset = ImageCaptionDataset(caption_path=caption_prediction_path.joinpath("validation-captions.csv"),
                                             images_dir=data_path.joinpath("validation-images", "valid"),
                                             sequence_length=200,
                                             text_tokenizer=tokenizer,
                                             text_transform=text_tranform)
    """
    uncomment when tesing locally to see how the model works 
    indices = list(range(0, 10))  # select your indices here as a list  
    train_subset = torch.utils.data.Subset(training_dataset, indices)
    validation_subset = torch.utils.data.Subset(validation_dataset, list(range(11, 30)))"""
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_dataloader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=validation_dataloader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                epoch=epoch)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint("images-captioning", epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    with mlflow.start_run():

        decoder.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()
        # Batches
        for i, (images, captions, lengths) in enumerate(iter(train_loader)):

            data_time.update(time.time() - start)

            # this part can be a function because it is repeated

            # Move to GPU, if available
            images = images.to(device)
            captions = captions.to(device)
            features = encoder(images) # batch size x hidden size or (512)
            
            features = features.expand(decoder_layers, features.shape[0], features.shape[1]) # (number of layer X batch size X hidden size)
            # Set the initial hidden state of the decoder to be the output of the encoder
            decoder_hidden = (features.contiguous(), features.contiguous())
            outputs = decoder(captions, decoder_hidden)
            # Calculate loss
            loss = criterion(outputs, captions)

            # Add doubly stochastic attention regularization
            # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean() # should comment to improve the model in the future

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()
            # Keep track of metrics
            # top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), 1)
            # top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses))
            step = epoch * len(train_loader) + i
            log_scalar('train_loss', losses.avg, step)


def validate(val_loader, encoder, decoder, criterion, epoch):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    with mlflow.start_run():
        decoder.eval()  # eval mode (no dropout or batchnorm)
        if encoder is not None:
            encoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()

        original_texts = list()  
        predicted_texts = list()
        

        # explicitly disable gradient calculation to avoid CUDA memory error
        # solves the issue #57


        # Batches

        with torch.no_grad():
            # Batches
            for i, (images, captions, lengths) in enumerate(iter(val_loader)):

                # Move to device, if available

                images = images.to(device)
                captions = captions.to(device)
                features = encoder(images)  # batch size x hidden size or (512)

                features = features.expand(decoder_layers, features.shape[0], features.shape[1]) # (number of layer X batch size X hidden size)
                # Set the initial hidden state of the decoder to be the output of the encoder
                decoder_hidden = (features.contiguous(), features.contiguous())
                outputs = decoder(captions, decoder_hidden)
                # Calculate loss
                loss = criterion(outputs, captions)
                predicted_indices = outputs.argmax(dim=1)

                assert predicted_indices.shape == captions.shape

                caption_text = get_text_for_batch(captions, vocabulary, lengths)
                original_texts.extend(caption_text)
                predicted_text = get_text_for_batch(predicted_indices, vocabulary, lengths)
                predicted_text = [" ".join(text) for text in predicted_text]
                predicted_texts.extend(predicted_text)
                # Keep track of metrics
                losses.update(loss.item(), sum(lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % print_freq == 0:
                    print('Validation: [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                          batch_time=batch_time,
                                                                          loss=losses))
                    for original, predicted in zip(original_texts[:5], predicted_texts[:5]):
                        print("original caption: {}".format(original))
                        print(20 * "=|=")
                        print("predicted caption: {}".format(predicted))
                        print(20 * "-----")

                assert len(original_texts) == len(predicted_texts)
            bleu4 = corpus_bleu(original_texts, predicted_texts,)
            print('\n * LOSS - {loss.avg:.3f},, BLEU-4 - {bleu}\n'.format(loss=losses, bleu=bleu4))
        step = (epoch + 1) * len(val_loader)
        log_scalar('test_loss', losses.avg, step)
        return bleu4


if __name__ == '__main__':
    main()
