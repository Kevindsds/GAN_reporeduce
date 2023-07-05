import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from gan_model import weights_init
from datasets import EEGDataset, ChbmitFolder
from torch.utils.data import DataLoader, Dataset
import os
from img_buffer import ImageHistoryBuffer

class SyntheticDataset(Dataset):
    def __init__(self, data_path):
        """
        :param data_path: path to the parent folder of synthetic data for the patient
        """
        data = os.listdir(data_path)
        data = [os.path.join(data_path, i) for i in data]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = self.data[index]
        mtx = np.load(fpath)
        return mtx

class Resnet_Block(nn.Module):
    """
    The "same" conv netblock, with residual connection retain original image
    information
    """
    def __init__(self, in_feature, k_size):
        """
        :param in_feature: # input channel
        :param k_size: Con2D kernel size
        """
        super(Resnet_Block, self).__init__()
        self.conv2d = nn.Conv2d(in_feature, in_feature, k_size, padding='same')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2d(x)
        out = self.relu(out)
        out = self.conv2d(out)
        out += x
        return self.relu(out)


class R_Generator(nn.Module):

    def __init__(self, img_channels, img_height, img_width):
        super(R_Generator, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        # TODO here're some feature variable could be modified
        # the number of feature for middle image
        n_feature = 64
        # resnet block kernel size
        k_size = 3
        self.input_layer = nn.Sequential(
            nn.Conv2d(img_channels, n_feature, k_size, padding="same"),
            nn.LeakyReLU(0.2),
        )
        self.resnet_block = Resnet_Block(n_feature, k_size)
        self.out_layer = nn.Sequential(
            nn.Conv2d(n_feature, img_channels, 1, padding='same'),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        out = self.input_layer(x)
        for _ in range(4):
            out = self.resnet_block(out)
        out = self.out_layer(out)
        return out
class R_Discriminator_2(nn.Module):

    def __init__(self, img_channels, img_height, img_width):
        super(R_Discriminator_2, self).__init__()
        self.latent_feature = 16
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.main_layer = nn.Sequential(
            # 22 x 128 x 32
            nn.Conv2d(img_channels, 32, kernel_size=3, padding='same'),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),

            # 96 x 64 x 16
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),

            #64, 32, 8
            nn.Conv2d(16, 8, kernel_size=3, padding='same'),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),

            #32 x 16 x 4
            nn.Conv2d(8, 4, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),

            #16 x 16 x 4
            nn.Conv2d(4, 2, kernel_size=1, padding='same')
            # 2 x 16 x 4
            # We keep the desciminator to output an down-sampled image because we want
            # local loss, not the global one for each part of the pixel
            # Besides, the channel count is 2 representing the confidence for [fake, real] channel
        )
        # self.main_layer = nn.Sequential(
        #     # 22 x 128 x 32
        #     nn.ZeroPad2d((1,2,1,2)),
        #     nn.Conv2d(img_channels, self.latent_feature, 5,2,0, bias=False),
        #     # 16 x 64 x 16
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(self.latent_feature, self.latent_feature//2, 5, padding='same', bias=False),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.BatchNorm2d(self.latent_feature //2),
        #     nn.LeakyReLU(0.2),
        #     # 8 x 32 x 8
        #     nn.Conv2d(self.latent_feature//2, self.latent_feature//4,5, padding='same', bias=False),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.BatchNorm2d(self.latent_feature // 4),
        #     nn.LeakyReLU(0.2),
        #     # 4 x 16 x 4
        #     nn.Conv2d(self.latent_feature // 4, self.latent_feature//8, 1, padding='same'),
        #     # 2 x 16 x 4
        # )

    def forward(self,x):
        return self.main_layer(x)
class R_Discriminator(nn.Module):

    def __init__(self, img_channels, img_height, img_width):
        super(R_Discriminator, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.main_layer = nn.Sequential(
            # 22 x 128 x 32
            nn.Conv2d(img_channels, 96, kernel_size=3, padding='same'),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),

            # 96 x 64 x 16
            nn.Conv2d(96, 64, kernel_size=3, padding='same'),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),

            #64, 32, 8
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),

            #32 x 16 x 4
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),

            #16 x 16 x 4
            nn.Conv2d(16, 2, kernel_size=1, padding='same')
            # 2 x 16 x 4
            # We keep the desciminator to output an down-sampled image because we want
            # local loss, not the global one for each part of the pixel
            # Besides, the channel count is 2 representing the confidence for [fake, real] channel
        )

    def forward(self, x):
        out = self.main_layer(x)
        # The output is of shape [batch size, 2, 16, 4], and we'll use cross entropy loss for calculation
        return out



class R_Train:
    def __init__(self, img_channels, img_height, img_width, data_loader_chb, data_loader_synthetic, learning_rate, device=None, DCGAN=None):
        self.device = torch.device('cpu')
            # ("cuda:0" if torch.cuda.is_available() and device == 'cuda' else "cpu")
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        # This is the GAN network to generate synthetic data, we can use it to substitute the synthetic dataloader
        self.DCGAN = DCGAN

        self.G = R_Generator(self.img_channels, img_height, img_width)
        self.D = R_Discriminator_2(self.img_channels, img_height, img_width)

        batch_size = data_loader_chb.batch_size
        self.img_buffer = ImageHistoryBuffer((0, self.img_channels, self.img_height, self.img_width), batch_size * 100, batch_size)

        # init the training functions
        self.loss_D = nn.CrossEntropyLoss()
        self.loss_G_img = nn.CrossEntropyLoss()
        self.loss_G_reg = nn.L1Loss()

        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=learning_rate)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=learning_rate)

        self.data_loader_chb = data_loader_chb
        self.data_iterator_chb = iter(self.data_loader_chb)
        self.data_loader_synthetic = data_loader_synthetic
        self.data_iterator_synthetic = iter(self.data_loader_synthetic) if data_loader_synthetic else None

    def iter_get_next(self, data_loader, data_iterator):
        """
        A helper function to return countless batch from the dataloader
        :param data_loader: the dataloader of the dataset
        :param data_iterator: the data iterator, generated from the dataloader
        :return:
        """
        try:
            data_out = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data_out = next(data_iterator)
        return data_out, data_iterator

    def get_next_real(self):
        """
        Get the next chb dataset batch
        :return:
        """
        data_chb, iter_chb = self.iter_get_next(self.data_loader_chb, self.data_iterator_chb)
        self.data_iterator_chb = iter_chb
        return data_chb[0]

    def get_next_synthetic(self, batch_size = None, condition =None):
        """
        Get the next synthetic dataset batch
        :param batch_size: only works when given the DCGAN network, which controls the output batchsize
        :return:
        """
        assert self.DCGAN is not None or self.data_loader_synthetic is not None, "We should at least have one synthetic loader"
        if self.DCGAN:
            if not batch_size:
                batch_size = self.data_loader_chb.batch_size
            if not condition:
                condition = np.random.randint(2, size=batch_size)
            data_synthetic = self.DCGAN.gen_synthetic(batch_size, condition)
        else:
            data_synthetic, iter_synthetic = self.iter_get_next(self.data_loader_synthetic, self.data_iterator_synthetic)
            self.data_iterator_synthetic = iter_synthetic
        return data_synthetic

    def gen_synthetic(self, batch, condition=None):
        """
        Generate the refined synthetic image of "batch" size
        :param batch: batch size for the output
        :return:
        """
        if not condition:
            condition = np.random.randint(2, size=batch)
        batch_synthetic = self.get_next_synthetic(batch, condition)
        refined_synthetic = self.G(batch_synthetic.to(device=self.device, dtype=torch.float32))
        return refined_synthetic.detach().cpu()

    def get_labels(self, batch_size):
        # This corresponds to the output from Descriminator, which is local estimate
        ones = torch.ones([batch_size, 1, self.img_height // 8, self.img_width // 8])
        zeros = torch.zeros([batch_size, 1, self.img_height // 8, self.img_width // 8])
        # real: [0, 1], fake: [1, 0]
        real_label = torch.concat([zeros, ones], dim=1)
        fake_label = torch.concat([ones, zeros], dim=1)
        return real_label, fake_label

    def confident_metric(self, D_out, is_real):
        """
        calculate D(x) (when is_real) / D(G(x)) (when !is_real)
        :param D_out: the output of D
        :param is_real: if D_in is real image
        :return:
        """
        softmax = nn.Softmax(dim=1)
        D_out = softmax(D_out)
        result = D_out[:,1, ...] if is_real else D_out[:, 0, ...]
        return torch.mean(result).item()

    def get_save_path(self, cache_dir):
        return os.path.join(cache_dir, "G"), os.path.join(cache_dir, "D")

    def pretrain(self, num_batch_G, num_batch_D, cache_dir=None):
        self.G.train()
        self.D.train()
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        self.G.to(self.device)
        self.D.to(self.device)

        # pretrain generator, by make it generate similar image as the source
        with tqdm(range(num_batch_G), desc='R_Generator Pretraining', leave=False) as G_tqdm:
            for _ in G_tqdm:
                batch = self.get_next_synthetic()
                batch = batch.to(self.device)
                self.opt_G.zero_grad()
                out_G = self.G(batch)
                loss_G = self.loss_G_reg(out_G, batch)
                loss_G.backward()
                self.opt_G.step()

                G_tqdm.set_postfix_str(f"G l1 diff: {loss_G}")

        with tqdm(range(num_batch_D), desc='R_Discriminator Pretraining', leave=False) as D_tqdm:
            for _ in D_tqdm:
                batch_real = self.get_next_real()
                batch_fake = self.get_next_synthetic()
                real_label, _ = self.get_labels(batch_real.shape[0])
                _, fake_label = self.get_labels(batch_fake.shape[0])
                batch_real = batch_real.to(self.device, dtype=torch.float32)
                batch_fake = batch_fake.to(self.device, dtype=torch.float32)
                real_label = real_label.to(self.device, dtype=torch.float32)
                fake_label = fake_label.to(self.device, dtype=torch.float32)


                self.opt_D.zero_grad()
                out_D_real = self.D(batch_real)
                loss_D_real = self.loss_D(out_D_real, real_label)
                out_D_fake = self.D(batch_fake)
                loss_D_fake = self.loss_D(out_D_fake, fake_label)

                loss_D = loss_D_fake + loss_D_real
                loss_D.backward()
                self.opt_D.step()

                D_tqdm.set_postfix_str(f"D Cross Entropy Loss {loss_D}")
        if cache_dir:
            path_G_state, path_D_state = self.get_save_path(cache_dir)
            torch.save(self.G.state_dict(), path_G_state)
            torch.save(self.D.state_dict(), path_D_state)

    def save_model(self, model, out_path):
        torch.save(model.state_dict(), out_path)

    def load_model(self, model_name, state_dict_path):
        """

        :param model_name: "D" or "G"
        :param state_dict_path: the path to the state dict
        :return:
        """
        assert model_name in ["D", "G"], "model name should be 'D' or 'G'"
        state_dict = torch.load(state_dict_path)
        if model_name == "D":
            self.D.load_state_dict(state_dict)
        if model_name == "G":
            self.G.load_state_dict(state_dict)

    def train(self, num_step, num_iter_G, num_iter_D, reg_lambda, pretrain=False, cache_dir=None, pretrain_G=200,
              pretrain_D=100):
        """
        :param num_step: The outer loop's number of iteration
        :param num_iter_G: Per each outer iteration, how many times do we udpate Generator
        :param num_iter_D: Per each outer iteration, how many times do we update Discriminator
        :param learning_rate:
        :return:
        """
        # init the models
        self.G.train()
        self.D.train()
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        self.G.to(self.device)
        self.D.to(self.device)

        # init the logging
        D_errors, G_errors = [], []
        D_xs, D_Z_xs = [], []

        # pretrain G and D
        if pretrain:
            path_G_state, path_D_state = self.get_save_path(cache_dir) if cache_dir else "", ""
            if os.path.exists(path_G_state) and os.path.exists(path_D_state):
                self.G.load_state_dict(torch.load(path_G_state))
                self.D.load_state_dict(torch.load(path_D_state))
            else:
                self.pretrain(pretrain_G, pretrain_D, cache_dir)

        with tqdm(range(num_step), desc='Refiner Model Training', leave=False) as GD_tqdm:
            for step in GD_tqdm:
                if step % 100 == 0 and cache_dir:
                    step_save_G = os.path.join(cache_dir, "G_ckpt")
                    step_save_D = os.path.join(cache_dir, "D_ckpt")
                    if not os.path.exists(step_save_D):
                        os.makedirs(step_save_D)
                    if not os.path.exists(step_save_G):
                        os.makedirs(step_save_G)
                    self.save_model(self.G, os.path.join(step_save_G, f"G_{step}"))
                    self.save_model(self.D, os.path.join(step_save_D, f"D_{step}"))

                # First train the refiner (Generator)
                for _ in range(num_iter_G):
                    batch_synthetic = self.get_next_synthetic()
                    batch_size = batch_synthetic.shape[0]
                    real_label, fake_label = self.get_labels(batch_size)
                    batch_synthetic = batch_synthetic.to(self.device, dtype=torch.float32)
                    real_label = real_label.to(self.device, dtype=torch.float32)
                    fake_label = fake_label.to(self.device, dtype=torch.float32)

                    self.opt_G.zero_grad()
                    G_out = self.G(batch_synthetic)
                    loss_reg  = self.loss_G_reg(G_out, batch_synthetic) * reg_lambda
                    D_out = self.D(G_out)
                    loss_discriminator = self.loss_G_img(D_out, real_label)
                    loss_G = loss_discriminator + loss_reg
                    loss_G.backward()
                G_errors.append(loss_G.item())

                # Then we train discriminator
                for _ in range(num_iter_D):
                    self.opt_D.zero_grad()
                    batch_synthetic = self.get_next_synthetic()
                    batch_size = batch_synthetic.shape[0]
                    batch_real = self.get_next_real()
                    _, fake_label = self.get_labels(batch_size)
                    real_label, _ = self.get_labels(batch_real.shape[0])
                    real_label = real_label.to(self.device, dtype=torch.float32)
                    fake_label = fake_label.to(self.device, dtype=torch.float32)
                    batch_synthetic = batch_synthetic.to(self.device, dtype=torch.float32)
                    batch_real = batch_real.to(self.device, dtype=torch.float32)

                    G_out = self.G(batch_synthetic)
                    G_out = G_out.detach()

                    # mix with the history buffer
                    G_out_cpu = G_out.cpu()
                    half_batch_from_image_history = self.img_buffer.get_from_image_history_buffer(batch_size // 2)
                    self.img_buffer.add_to_image_history_buffer(G_out_cpu.numpy())
                    if len(half_batch_from_image_history):
                        G_out[:batch_size // 2] = torch.from_numpy(half_batch_from_image_history).to(self.device, dtype=torch.float32)

                    D_out_real = self.D(batch_real)
                    D_out_fake = self.D(G_out)
                    D_x = self.confident_metric(D_out_real, True)
                    D_Z_x = self.confident_metric(D_out_fake, False)

                    loss_real = self.loss_D(D_out_real, real_label)
                    loss_fake = self.loss_D(D_out_fake, fake_label)
                    loss_D = loss_real + loss_fake
                    loss_D.backward()
                    self.opt_D.step()
                D_errors.append(loss_D.item())
                D_xs.append(D_x)
                D_Z_xs.append(D_Z_x)
                GD_tqdm.set_postfix_str(f"G error: {loss_G.item()}, D error: {loss_D.item()}, D(x): {D_x}, D(Z(x)): {D_Z_x}")
        return G_errors, D_errors, D_xs, D_Z_xs

if __name__ == "__main__":
    # refiner = R_Generator(128, 32, 22)
    # a = torch.rand(64, 22, 128, 32)
    # print(refiner(a).shape)
    # descriminator = R_Descriminator(128, 32, 22)
    # print(descriminator(a).shape)

    mapping = {
        'preictal_1': 0,
        'interictal': 1,
    }

    chbmit = ChbmitFolder("/home/tian/DSI/datasets/CHB-MIT-log", "train", mapping, True)
    dataset_chb = EEGDataset(chbmit.get_all_data())
    # chb_dataloader = DataLoader(chb_dataset, batch_size=1, shuffle=True)
    data_loader_chb = DataLoader(dataset_chb, batch_size=64, shuffle=True)
    dataset_synthetic = SyntheticDataset("/home/tian/DSI/datasets/CHB_synthetic")
    data_loader_synthetic = DataLoader(dataset_synthetic, batch_size=64, shuffle=True)

    rtrain = R_Train(22, 128, 32, data_loader_chb, data_loader_synthetic, 0.0001, device='cuda')
    rtrain.pretrain(1, 1)
    rtrain.train(1000, 1,1,0.1, pretrain=False)


