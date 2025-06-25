import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import wandb
import os
from tqdm import tqdm
from dataLoaders import get_dataloaders
from models import Generator, Discriminator
from encoder import FaceNetEncoder

config = {
    "project_name": "conditional-face-gan-pytorch", 
    "run_name": f"self-attention-run-widerface-bs32-epochs20-stabilized", 
    "num_epochs": 20,
    "batch_size": 32,
    "image_size": 128,
    "noise_dim": 100,
    "embedding_dim": 512,
    "g_lr": 0.0002,
    "d_lr": 0.0001, #  Slower learning rate for the Discriminator
    "beta1": 0.5,
    "beta2": 0.999,
    "log_interval": 50,
    "sample_interval": 200,
    "checkpoint_dir": "./checkpoints_widerface",
    "log_file": "training_log.txt",
}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def evaluate_epoch(generator, discriminator, encoder, test_loader, criterion, device):
    generator.eval()
    discriminator.eval()
    total_g_loss = 0
    total_d_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set", leave=False):
            real_images = batch[0].to(device)
            current_batch_size = real_images.size(0)
            real_embeddings = encoder(real_images)
            output_real = discriminator(real_images, real_embeddings)
            labels_real = torch.full((current_batch_size,), 0.9, device=device) # Label smoothing
            loss_d_real = criterion(output_real, labels_real)

            noise = torch.randn(current_batch_size, config["noise_dim"], device=device)
            fake_images = generator(noise, real_embeddings)
            output_fake = discriminator(fake_images, real_embeddings)
            labels_fake = torch.full((current_batch_size,), 0.0, device=device)
            loss_d_fake = criterion(output_fake, labels_fake)
            
            total_d_loss += (loss_d_real + loss_d_fake).item()

            output_g = discriminator(fake_images, real_embeddings)
            loss_g = criterion(output_g, labels_real) # Fool the discriminator
            total_g_loss += loss_g.item()

    avg_g_loss = total_g_loss / len(test_loader)
    avg_d_loss = total_d_loss / len(test_loader)
    
    generator.train() # Set back to training mode
    discriminator.train()
    
    return avg_g_loss, avg_d_loss


def train():
    wandb.init(
        project=config["project_name"], 
        name=config["run_name"], 
        config=config
    )
    
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    with open(config["log_file"], "w") as f:
        f.write("Epoch,Train_G_Loss,Train_D_Loss,Test_G_Loss,Test_D_Loss\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    train_loader, test_loader = get_dataloaders(data_root='.', batch_size=config["batch_size"])
    if not train_loader:
        print("Could not create dataloaders. Exiting.")
        return


    encoder = FaceNetEncoder(device=device)
    generator = Generator(noise_dim=config["noise_dim"], embedding_dim=config["embedding_dim"]).to(device)
    discriminator = Discriminator(embedding_dim=config["embedding_dim"]).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=config["g_lr"], betas=(config["beta1"], config["beta2"]))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["d_lr"], betas=(config["beta1"], config["beta2"]))


    fixed_noise = torch.randn(config["batch_size"], config["noise_dim"], device=device)
    fixed_batch_for_samples = next(iter(test_loader))
    fixed_sample_images = fixed_batch_for_samples[0].to(device)
    with torch.no_grad():
        fixed_embeddings = encoder(fixed_sample_images)

    print("--- Starting Training ---")
    for epoch in range(config["num_epochs"]):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for i, batch in enumerate(pbar):
            real_images = batch[0].to(device)
            current_batch_size = real_images.size(0)

            if current_batch_size != config["batch_size"]:
                continue

            #  Train Discriminator 
            discriminator.zero_grad()
            with torch.no_grad():
                real_embeddings = encoder(real_images)
            
            output_real = discriminator(real_images, real_embeddings)
            labels_real = torch.full((current_batch_size,), 0.9, device=device)
            loss_d_real = criterion(output_real, labels_real)
            loss_d_real.backward()
            
            noise = torch.randn(current_batch_size, config["noise_dim"], device=device)
            fake_images = generator(noise, real_embeddings)
            output_fake = discriminator(fake_images.detach(), real_embeddings)
            labels_fake = torch.full((current_batch_size,), 0.0, device=device)
            loss_d_fake = criterion(output_fake, labels_fake)
            loss_d_fake.backward()
            
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            
            # Train Generator 
            generator.zero_grad()
            output_g = discriminator(fake_images, real_embeddings)
            loss_g = criterion(output_g, labels_real)
            loss_g.backward()
            optimizer_g.step()

            # Logging and Progress Bar
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            if i % config["log_interval"] == 0:
                print(f"\n[Epoch {epoch+1}/{config['num_epochs']}] [Batch {i}/{len(train_loader)}] G_Loss: {loss_g.item():.4f}, D_Loss: {loss_d.item():.4f}")
                
                pbar.set_postfix({
                    "G_loss": f"{loss_g.item():.4f}", 
                    "D_loss": f"{loss_d.item():.4f}"
                })
                wandb.log({
                    "Batch Generator Loss": loss_g.item(), 
                    "Batch Discriminator Loss": loss_d.item()
                })


        avg_train_g_loss = epoch_g_loss / len(train_loader)
        avg_train_d_loss = epoch_d_loss / len(train_loader)
        
        avg_test_g_loss, avg_test_d_loss = evaluate_epoch(generator, discriminator, encoder, test_loader, criterion, device)
        
        # This print provides a summary at the end of the epoch
        print(f"\n--- End of Epoch {epoch+1} Summary ---")
        print(f"  Avg Train Loss -> G: {avg_train_g_loss:.4f}, D: {avg_train_d_loss:.4f}")
        print(f"  Avg Test Loss  -> G: {avg_test_g_loss:.4f}, D: {avg_test_d_loss:.4f}\n")
        
        wandb.log({
            "Epoch": epoch + 1,
            "Avg Train Generator Loss": avg_train_g_loss,
            "Avg Train Discriminator Loss": avg_train_d_loss,
            "Avg Test Generator Loss": avg_test_g_loss,
            "Avg Test Discriminator Loss": avg_test_d_loss,
        })

        with torch.no_grad():
            generated_samples = generator(fixed_noise, fixed_embeddings).detach().cpu()
        wandb.log({
            "Generated Samples": wandb.Image(make_grid(generated_samples, normalize=True)),
            "Real Samples": wandb.Image(make_grid(fixed_sample_images, normalize=True))
        })
        
        with open(config["log_file"], "a") as f:
            f.write(f"{epoch+1},{avg_train_g_loss:.4f},{avg_train_d_loss:.4f},{avg_test_g_loss:.4f},{avg_test_d_loss:.4f}\n")

        torch.save(
            generator.state_dict(), 
            os.path.join(config["checkpoint_dir"], f"generator_epoch_{epoch+1}.pth")
        )
        torch.save(
            discriminator.state_dict(), 
            os.path.join(config["checkpoint_dir"], f"discriminator_epoch_{epoch+1}.pth")
        )
        
    print("--- Training Finished ---")
    wandb.finish()


if __name__ == '__main__':
    train()
