"""
The conditional DCGAN Model

The structure of the main hidden layer is borrowed from this paper:
https://ieeexplore.ieee.org/document/8853232

Reference: N. D. Truong, L. Kuhlmann, M. R. Bonyadi, D. Querlioz, L. Zhou and O. Kavehei, "Epileptic Seizure
Forecasting With Generative Adversarial Networks," in IEEE Access, vol. 7, pp. 143999-144009, 2019,
doi: 10.1109/ACCESS.2019.2944691.


Comment on dimensions:
The current generated CHB-MIT data is of shape [n-channel, frequency, time], for exampel: [22, 128, 32]
The current image in-take considered by pytorch is of shape [Batchsize, n-channel, frequency, time]
"""
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

# ============ utils ===========================
def weights_init(m):
    """
    Randomly init the network param
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Embed') != -1:
        nn.init.uniform_(m.weight.data, -1.0, 1.0)


#============== networks =======================
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Generator(nn.Module):
    """
    Quote from the original paper about the structure of Generator:
        The Generator takes a 100 dimensional sample from a uniform distribution U(−1,1) as input. The input is fully-
        connected with a hidden layer with the output size of 6272 which is then reshaped to 64×7×14 . The hidden layer
        is followed by three de-convolution layers with filter size 5×5 , stride 2×2 . Number of filters of the three de-
        convolution layers are 32, 16 and n , respectively. Outputs of the Generator have the same dimension with STFT of
        28 seconds EEG signals.
    """

    def __init__(self, output_channels, num_classes, noise_dim):
        """
        :param output_channels: the number of channels of the synthetic EEG
        :param num_classes: the number of eeg types (like preictal, interictal, ...)
        :param noise_dim: the dimension of the noise
        """
        super(Generator, self).__init__()
        self.noise_latent_feature = 64
        self.condition_latent_feature = 64
        self.total_num_feature = self.noise_latent_feature + self.condition_latent_feature

        self.latent_width = 2
        self.latent_height = 8
        self.noise_layer = nn.Sequential(
            nn.Linear(noise_dim, self.latent_width * self.latent_height * self.noise_latent_feature, bias=False),
            Reshape([-1, self.noise_latent_feature, self.latent_height, self.latent_width])
        )
        self.condition_layer = nn.Sequential(
            nn.Linear(num_classes, self.latent_width * self.latent_height * self.condition_latent_feature, bias=False),
            Reshape([-1, self.condition_latent_feature, self.latent_height, self.latent_width])
        )

        self.main_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            # input shape 128x2x8
            nn.ConvTranspose2d(self.total_num_feature, self.total_num_feature // 2, 5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(self.total_num_feature // 2),
            nn.LeakyReLU(0.2),
            # output shape 64x4x16
            nn.ConvTranspose2d(self.total_num_feature // 2, self.total_num_feature // 4, 5, stride=2, padding=2,
                               output_padding=1,
                               bias=False),
            nn.BatchNorm2d(self.total_num_feature // 4),
            nn.LeakyReLU(0.2),
            # output shape 32x8x32
            nn.ConvTranspose2d(self.total_num_feature // 4, self.total_num_feature // 8, 5, stride=2, padding=2,
                               output_padding=1,
                               bias=False),
            nn.BatchNorm2d(self.total_num_feature // 8),
            nn.LeakyReLU(0.2),
            # output shape 16x16x64
        )

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(self.total_num_feature // 8, output_channels, 5, stride=2, padding=2, output_padding=1,
                               bias=False),
            # output shape 3x32x128
        )

    def forward(self, noise, condition):
        """
        :param noise: shape [batch_size, noise_dim]
        :param condition: one hot vector, of shape [batch_size, num_class]
        :return:
        """
        h_noise = self.noise_layer(noise)
        h_cond = self.condition_layer(condition)
        x = torch.cat([h_noise, h_cond], dim=1)
        h = self.main_layer(x)
        out = self.output_layer(h)
        return out


class Discriminator(nn.Module):
    """
    Quote from the original paper about the structure of Discriminator:
        The Discriminator, on the other hand, is configured to discriminate the generated EEG signals from the original
        ones. The Discriminator consists of three convolution layers with filter size 5×5 , stride 2×2 . Number of filters
        of the three convolution layers are 16, 32 and 64, respectively.
    """

    def __init__(self, input_channels, num_classes):
        super(Discriminator, self).__init__()
        self.data_latent_feature = 8
        self.condition_latent_feature = 8
        self.total_num_feature = self.data_latent_feature + self.condition_latent_feature

        self.data_layer = nn.Sequential(
            # input shape 3x32x128
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(input_channels, self.data_latent_feature, 5, 2, 0, bias=False),
        )
        self.condition_layer = nn.Sequential(
            # input shape 2x32x128
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(num_classes, self.condition_latent_feature, 5, 2, 0, bias=False),
        )

        self.main_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            # input shape 16x16x64
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(self.total_num_feature, self.total_num_feature * 2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(self.total_num_feature * 2),
            nn.LeakyReLU(0.2),
            # output shape 32x8x32
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(self.total_num_feature * 2, self.total_num_feature * 4, 5, 2, 0, bias=False),
            nn.BatchNorm2d(self.total_num_feature * 4),
            nn.LeakyReLU(0.2),
            # output shape 64x4x16
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(self.total_num_feature * 4, self.total_num_feature * 8, 5, 2, 0, bias=False),
            nn.BatchNorm2d(self.total_num_feature * 8),
            nn.LeakyReLU(0.2),
            # output shape 128x2x8
            nn.Flatten()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.total_num_feature * 8 * 2 * 8, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, data, condition):
        """
        :param data: [3, 32, 128]
        :param condition: [2, 32, 128]
        :return:
        """
        h_data = self.data_layer(data)
        h_cond = self.condition_layer(condition)
        x = torch.cat([h_data, h_cond], dim=1)
        h = self.main_layer(x)
        out = self.output_layer(h)
        return out


class CDCGAN:

    def __init__(self, data_channels, num_classes, noise_dim=100, device='cpu'):
        self.device = torch.device(device)

        self.noise_dim = noise_dim
        self.G = Generator(data_channels, num_classes, noise_dim)
        self.D = Discriminator(data_channels, num_classes)

        self.label_dim = num_classes
        # an array of <num_classes> one hot vectors
        self.onehot = nn.functional.one_hot(torch.arange(self.label_dim)).type(torch.float32)

        # This is to generate the one-hot version as a image input that could be later concatenated with the
        # image.
        # eg: if the label is 1, then self.fill[1] only has the index 1 as 1, while others are all 0
        self.fill = torch.zeros([self.label_dim, self.label_dim, 128, 32])
        for i in range(self.label_dim):
            self.fill[i, i, :, :] = 1

    def gen_latent_noise(self, batch):
        """Generate noise from distribution Uniform(-1,1)"""
        return torch.rand(batch, self.noise_dim) * -2 + 1

    def train(self, train_data, num_epochs, learning_rate, betas=(0.5, 0.999), logging_on=True):

        self.G.to(self.device)
        self.D.to(self.device)

        self.G.train()
        self.D.train()

        G_optimizer = torch.optim.Adam(self.G.parameters(), lr=learning_rate, betas=betas)
        D_optimizer = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=betas)

        # TODO: Maybe change to other better loss functions later
        loss_criterion = nn.BCELoss()

        G_losses = []
        D_losses = []

        for ep in tqdm(range(num_epochs)):
            for i, (data_real, labels_real) in enumerate(train_data):
                labels_real = labels_real.type(torch.long)
                data_real = data_real.type(torch.float32)

                batch = data_real.size()[0]

                # Step 1. Feed real data to train Discriminator
                # Assume labels come in as a list/tensor/array of integers: e.g. [0,1,1,0,1]
                conditions_real = self.fill[labels_real]
                decision_real = self.D(data_real.to(self.device), conditions_real.to(self.device)).squeeze()
                all_real = torch.ones(batch)
                D_loss_real = loss_criterion(decision_real, all_real.to(self.device))

                # Step 2. Generate fake data from Generator
                # labels_fake = (torch.rand(batch, 1) * self.label_dim).type(torch.LongTensor).squeeze()
                labels_fake = labels_real
                conds_fake_onehot = self.onehot[labels_fake]
                # Generate noise as inputs for Generator
                noise = self.gen_latent_noise(batch)
                data_fake = self.G(noise.to(self.device), conds_fake_onehot.to(self.device))

                # Step 3. Feed fake data to train Discriminator
                conditions_fake = self.fill[labels_fake]
                decision_fake = self.D(data_fake.to(self.device), conditions_fake.to(self.device)).squeeze()
                all_fake = torch.zeros(batch)
                D_loss_fake = loss_criterion(decision_fake, all_fake.to(self.device))

                # Step 4. Compute total loss of Discriminator and update weights
                D_loss = D_loss_fake + D_loss_real
                self.D.zero_grad()
                D_loss.backward()
                D_optimizer.step()

                # Step 5. Compute loss of Generator
                data_fake = self.G(noise.to(self.device), conds_fake_onehot.to(self.device))
                decision_fake = self.D(data_fake.to(self.device), conditions_fake.to(self.device)).squeeze()
                G_loss = loss_criterion(decision_fake, all_real.to(self.device))
                self.G.zero_grad()
                G_loss.backward()
                G_optimizer.step()

                # (Optional) Step 6. update Generator for the second time
                data_fake = self.G(noise.to(self.device), conds_fake_onehot.to(self.device))
                decision_fake = self.D(data_fake.to(self.device), conditions_fake.to(self.device)).squeeze()
                G_loss = loss_criterion(decision_fake, all_real.to(self.device))
                self.G.zero_grad()
                G_loss.backward()
                G_optimizer.step()

                G_losses.append(G_loss.detach().cpu())
                D_losses.append(D_loss.detach().cpu())

                if logging_on:
                    print(
                        f"Epoch {ep + 1}/{num_epochs} | Batch {i + 1}/{len(train_data)} :\tD Loss:{D_loss}\t G Loss:{G_loss}")
        if logging_on:
            print("Done.")
        return G_losses, D_losses

    def generate(self, conditions, noises=None):
        if noises is None:
            noises = self.gen_latent_noise(len(conditions))
        with torch.no_grad():
            conds = torch.Tensor(self.onehot[conditions])
            return self.G(noises.to(self.device), conds.to(self.device)).cpu()

    def save_generator(self, model_path):
        torch.save(self.G.state_dict(), model_path)

    def save_discriminator(self, model_path):
        torch.save(self.D.state_dict(), model_path)

    def load_generator(self, model_path):
        self.G.load_state_dict(torch.load(model_path))

    def load_discriminator(self, model_path):
        self.D.load_state_dict(torch.load(model_path))

class Generator_i(nn.Module):
    """
    Improve the generator embedding part by directly using the embedding, rather than the one hot vector as input
    """
    def __init__(self, output_channels, num_classes, noise_dim):
        """
        :param output_channels: the number of channels of the synthetic EEG
        :param num_classes: the number of eeg types (like preictal, interictal, ...)
        :param noise_dim: the dimension of the noise
        """
        super(Generator_i, self).__init__()
        self.noise_latent_feature = 64
        self.condition_latent_feature = 64
        self.total_num_feature = self.noise_latent_feature + self.condition_latent_feature

        self.latent_width = 2
        self.latent_height = 8

        self.noise_layer = nn.Sequential(
            nn.Linear(noise_dim, self.latent_width * self.latent_height * self.noise_latent_feature, bias=False),
            Reshape([-1, self.noise_latent_feature, self.latent_height, self.latent_width])
        )
        self.condition_layer = nn.Sequential(
            nn.Embedding(num_classes, self.latent_width * self.latent_height * self.condition_latent_feature),
            Reshape([-1, self.condition_latent_feature, self.latent_height, self.latent_width])
        )

        self.main_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            # input shape 128x8x2
            nn.ConvTranspose2d(self.total_num_feature, self.total_num_feature // 2, 5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(self.total_num_feature // 2),
            nn.LeakyReLU(0.2),
            # output shape 64x16x4
            nn.ConvTranspose2d(self.total_num_feature // 2, self.total_num_feature // 4, 5, stride=2, padding=2,
                               output_padding=1,
                               bias=False),
            nn.BatchNorm2d(self.total_num_feature // 4),
            nn.LeakyReLU(0.2),
            # output shape 32x32x8
            nn.ConvTranspose2d(self.total_num_feature // 4, self.total_num_feature // 8, 5, stride=2, padding=2,
                               output_padding=1,
                               bias=False),
            nn.BatchNorm2d(self.total_num_feature // 8),
            nn.LeakyReLU(0.2),
            # output shape 16x64x16
        )

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(self.total_num_feature // 8, output_channels, 5, stride=2, padding=2, output_padding=1,
                               bias=False),
            # output shape 3x128x32
        )

    def forward(self, noise, condition):
        """
        :param noise: shape [batch_size, noise_dim]
        :param condition: the 1d array of size "batch_size"
        :return:
        """
        h_noise = self.noise_layer(noise)
        condition = condition.to(torch.int32)
        h_cond = self.condition_layer(condition)
        x = torch.cat([h_noise, h_cond], dim=1)
        h = self.main_layer(x)
        out = self.output_layer(h)
        return out

class Discriminator_i(nn.Module):
    """
    Improved descriminator, by adding embedding
    """
    def __init__(self, input_channels, num_classes, device=None):
        """
        :param input_channels: the number of channels for the input synthetic data
        :param num_classes: the number of seizure classes we're trying to identify
        :param input_height: the height of the input (time)
        :param input_width: the width of the input (frequency)
        """
        self.device = torch.device('cpu')
        # if not device and torch.cuda.is_available():
        #     self.device = 'cuda'
        super(Discriminator_i, self).__init__()
        self.data_latent_feature = 8
        self.condition_latent_feature = 8
        self.num_classes = num_classes
        self.total_num_feature = self.data_latent_feature + self.condition_latent_feature

        self.data_layer = nn.Sequential(
            # input shape 3x32x128
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(input_channels, self.data_latent_feature, 5, 2, 0, bias=False),
        )

        self.main_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            # input shape 16x16x64
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(self.total_num_feature, self.total_num_feature * 2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(self.total_num_feature * 2),
            nn.LeakyReLU(0.2),
            # output shape 32x8x32
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(self.total_num_feature * 2, self.total_num_feature * 4, 5, 2, 0, bias=False),
            nn.BatchNorm2d(self.total_num_feature * 4),
            nn.LeakyReLU(0.2),
            # output shape 64x4x16
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(self.total_num_feature * 4, self.total_num_feature * 8, 5, 2, 0, bias=False),
            nn.BatchNorm2d(self.total_num_feature * 8),
            nn.LeakyReLU(0.2),
            # output shape 128x2x8
            nn.Flatten()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.total_num_feature * 8 * 2 * 8, 1, bias=False),
            nn.Sigmoid(),
        )

    def get_condition_layer(self, feature_dim=None):
        if not feature_dim:
            feature_dim =  [self.condition_latent_feature ,16 ,64]
        layer = nn.Sequential(
            # input shape is [1]
            nn.Embedding(self.num_classes, np.prod(feature_dim)),
            Reshape([-1, *feature_dim])
        )
        layer.to(self.device)
        return layer

    def forward(self, data, condition):
        """
        :param data: [3, 32, 128]
        :param condition: [1]
        :return:
        """
        h_data = self.data_layer(data)
        self.condition_layer = self.get_condition_layer(h_data.shape[1:])
        condition = condition.to(torch.int32)
        h_cond = self.condition_layer(condition)
        x = torch.cat([h_data, h_cond], dim=1)
        h = self.main_layer(x)
        out = self.output_layer(h)
        return out

class DCGAN_i:
    """
    This is the improved verison of the GAN network above
    """
    def __init__(self, data_channels, num_classes, noise_dim=100, device='cpu'):
        self.device = torch.device('cpu')
            # ("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
        self.data_channels = data_channels
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.G = Generator_i(data_channels, num_classes, noise_dim)
        self.D = Discriminator_i(data_channels, num_classes)

    def gen_latent_noise(self, batch):
        """Generate noise from distribution Uniform(0,1)"""
        return torch.rand(batch, self.noise_dim)

    def gen_synthetic(self, batch, condition):
        """
        Method to generate a batch size of synthetic data
        :param batch: the batch size for the output synthetic data
        :param condition: np array of batch size, either 0 or 1
        :return:
        """
        noise = self.gen_latent_noise(batch)
        condition = torch.tensor(condition, dtype=torch.int32, device=self.device)
        synthetic = self.G(noise.to(device=self.device, dtype=torch.float32), condition)
        synthetic = synthetic.detach().cpu()
        return synthetic

    def train(self, data_loader, num_epochs, learning_rate, adam_beta=(0.5, 0.999)):
        # init the models
        self.G.train()
        self.D.train()
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        self.G.to(self.device)
        self.D.to(self.device)

        # init the training functions
        loss_criterion = nn.BCELoss()
        opt_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate, betas=adam_beta)
        opt_G = torch.optim.Adam(self.G.parameters(), lr = learning_rate, betas=adam_beta)
        real_label = 1
        fake_label = 0

        # init the logging
        D_errors, G_errors = [], []
        D_xs, D_Z_xs = [], []


        for epoch in range(num_epochs):
            with tqdm(data_loader, unit='batch', leave=False) as tepoch:
                for eeg_data, seizure_type in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    b_size = seizure_type.shape[0]
                    real_labels = torch.full((b_size,1), real_label, device=self.device, dtype=torch.float32)
                    fake_labels = torch.full((b_size,1), fake_label, device=self.device, dtype=torch.float32)

                    # Update the Descriminator, first with real batches:
                    eeg_data = eeg_data.to(self.device, dtype=torch.float32)
                    seizure_type = seizure_type.to(self.device, dtype=torch.int32)
                    self.D.zero_grad()
                    D_out_real = self.D(eeg_data, seizure_type)
                    D_error_real = loss_criterion(D_out_real, real_labels)
                    D_error_real.backward()
                        # The average should go from high to low (to approximately 0.5)
                    D_out_real_avg = D_out_real.mean().item()
                    D_xs.append(D_out_real_avg)

                    # Update the Descriminator, with the fake batches:
                    noise = torch.randn(b_size, self.noise_dim).to(self.device)
                    G_out_fake = self.G(noise, seizure_type)
                    D_out_fake = self.D(G_out_fake.detach(), fake_labels)
                    D_error_fake = loss_criterion(D_out_fake, fake_labels)
                    D_error_fake.backward()

                    # Sum both real and fake because loss function is -log(D(x)) + -log(1-D(Z(x)))
                    D_error = D_error_fake + D_error_real
                    opt_D.step()

                    # Update Generator
                    self.G.zero_grad()
                    # Use the updated D to re-compute the output
                    D_out_fake = self.D(G_out_fake, fake_labels)
                        # G wants to max -log(1-D(Z(x))), which is equivelent to min -log(D(Z(x))
                    G_error = loss_criterion(D_out_fake, real_labels)
                    G_error.backward()
                    opt_G.step()
                    D_out_fake_avg = D_out_fake.mean().item()
                    D_Z_xs.append(D_out_fake_avg)

                    D_errors.append(D_error.item())
                    G_errors.append(G_error.item())

                    tepoch.set_postfix_str(f"G error: {G_error}, D error: {D_error}, D(x): {D_out_real_avg}, D(Z(x)): {D_out_fake_avg}")
        return D_errors, G_errors, D_xs, D_Z_xs

if __name__ == "__main__":
    import datasets
    from torch.utils.data import DataLoader
    mapping = {
        'preictal_1': 0,
        'interictal': 1,
    }

    chbmit = datasets.ChbmitFolder("/home/tian/DSI/datasets/CHB-MIT-log", "train", mapping, True)
    chb_dataset = datasets.EEGDataset(chbmit.get_all_data())
    # chb_dataloader = DataLoader(chb_dataset, batch_size=1, shuffle=True)
    chb_dataloader = DataLoader(chb_dataset, batch_size=64, shuffle=True)

    dcgan_i = DCGAN_i(22, 2)
    dcgan_i.train(chb_dataloader, 5, 0.1)
