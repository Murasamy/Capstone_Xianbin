import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

kernel_size = (5, 3)#(5, 3) # (4, 4) kernel
stride = (1, 1)
# dilation = (2, 1)
padding = (2, 1)
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling
kernel_size_decoder = (3, 3)
stride_decoder = (1, 1)
padding_decoder = (1, 1)


# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ConvVAE, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=stride, padding=padding
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=stride, padding=0
        )

        # decoder_inputsize_calculator
        self.dec_input_4 = nn.Conv2d(in_channels=image_channels, out_channels=init_channels*2, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=padding_decoder)
        self.dec_input_3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=padding_decoder)
        self.dec_input_2 = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=padding_decoder)
        self.dec_input_1 = nn.Conv2d(in_channels=init_channels*8, out_channels=64, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=0)

        # fully connected layers for learning representations
        encoder_outsize, self.decoder_inputsize = self._get_conv_out()
        # print("decoder input", encoder_outsize, self.decoder_inputsize)

        self.fc1 = nn.Linear(int(np.prod(encoder_outsize)), 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, int(np.prod(self.decoder_inputsize)))
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=padding_decoder
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=padding_decoder
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size_decoder, 
            stride=stride_decoder, padding=padding_decoder
        )
        
    def _get_conv_out(self):
        o = torch.zeros(1, *self.input_shape)
        o = F.relu(self.enc1(o))
        o = F.relu(self.enc2(o))
        o = F.relu(self.enc3(o))
        o = F.relu(self.enc4(o))
        encoder_output_size = o.size()
        o = torch.zeros(1, *self.output_shape)
        o = F.relu(self.dec_input_4(o))
        o = F.relu(self.dec_input_3(o))
        o = F.relu(self.dec_input_2(o))
        o = F.relu(self.dec_input_1(o))
        decoder_input_size = o.size()
        return encoder_output_size, decoder_input_size

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        # print("encode 1 shape:", x.shape)
        x = F.relu(self.enc2(x))
        # print("encode 2 shape:", x.shape)
        x = F.relu(self.enc3(x))
        # print("encode 3 shape:", x.shape)
        x = F.relu(self.enc4(x))
        # print("encode 4 shape:", x.shape)
        # print("shape last layer of encoder", x.shape)
        # print("shape x 1:", x.shape)
        batch, _, h_eo, w_eo = x.shape
        
        x = F.adaptive_avg_pool2d(x, (h_eo, w_eo)).reshape(batch, -1)
        # print("shape x 2:", x.shape)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        # z = z.view(-1, 64, 1, 1)
        # print("shape z:", z.shape)
        z = z.view(-1, self.decoder_inputsize[0], self.decoder_inputsize[1], self.decoder_inputsize[2]) #the shape of z should be the input_size of the decoder
        # decoding
        x = F.relu(self.dec1(z))
        # print("decode 1 shape:", x.shape)
        x = F.relu(self.dec2(x))
        # print("decode 2 shape:", x.shape)
        x = F.relu(self.dec3(x))
        # print("decode 3 shape:", x.shape)
        reconstruction = torch.sigmoid(self.dec4(x))
        # print("decode 4 shape:", reconstruction.shape)
        return reconstruction, mu, log_var