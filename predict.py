import torch 
from pathlib import Path
from src.data.dataloader import ImageCaptionDataset

checkpoint_path = Path.cwd().joinpath("checkpoints",
                                      "lstm-8-layers-without-beam",
                                      "BEST_checkpoint_images-captioning.pth.tar")

caption_prediction_path = Path.cwd().joinpath("data", "caption-prediction")
validation_dataset = ImageCaptionDataset(caption_path=caption_prediction_path.joinpath("validation-captions.csv"),
                                         images_dir=data_path.joinpath("validation-images", "valid"),
                                         sequence_length=200,
                                         text_tokenizer=tokenizer,
                                         text_transform=text_tranform)

checkpoint = torch.load(checkpoint_path)
start_epoch = checkpoint['epoch'] + 1
epochs_since_improvement = checkpoint['epochs_since_improvement']
best_bleu4 = checkpoint['bleu-4']
decoder = checkpoint['decoder']
decoder_optimizer = checkpoint['decoder_optimizer']
encoder = checkpoint['encoder']
encoder_optimizer = checkpoint['encoder_optimizer']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

decoder = decoder.to(device)
encoder = encoder.to(device)
