import torch
import torch.nn as nn
import torch.nn.functional as F
import hmac
import hashlib
from copy import deepcopy


class HufuNet:
    def __init__(self, secret_key="2020"):
        self.secret_key = secret_key

    def embed(self, init_model, encoder_state_dict, decoder_state_dict, config):
        """
        Embeds the encoder parameters of HufuNet into a DNN model

        Args:
            init_model: The model to embed watermark into
            encoder_state_dict: State dict of HufuNet's encoder
            decoder_state_dict: State dict of HufuNet's decoder (used for hashing)
            config: Configuration dict including watermark_size, etc.

        Returns:
            Watermarked model
        """
        # Create a deep copy of the model to avoid modifying the original
        model_watermarked = deepcopy(init_model)

        # Extract all parameters from the model to create a flat parameter vector
        all_params = []
        layers_info = {}

        for name, param in model_watermarked.named_parameters():
            if 'conv' in name and 'weight' in name:
                # Create a flat view of each convolution layer's parameters
                for i in range(param.size(0)):
                    param_flat = param[i].view(-1)
                    all_params.append(param_flat)

                    # Store information about layer dimensions
                    if name not in layers_info:
                        layers_info[name] = {
                            'shape': param[0].size(),
                            'count': param.size(0)
                        }

        # Flatten all params into a single vector
        param_vector = torch.cat(all_params)
        total_params = param_vector.size(0)

        # Create a bitmap to keep track of the used positions
        bitmap = torch.zeros(total_params, dtype=torch.bool)

        # Get encoder parameters to embed
        encoder_params = []
        for name, param in encoder_state_dict.items():
            if 'weight' in name:
                # Handle each parameter tensor by flattening it
                encoder_params.append(param.view(-1))

        # Flatten all encoder parameters
        encoder_vector = torch.cat(encoder_params)
        watermark_size = encoder_vector.size(0)

        # Embed each parameter of the encoder using the hash function
        for i in range(watermark_size):
            # Get the mirrored parameter from the decoder
            # We get the corresponding decoder parameter by transforming the name
            decoder_param_value = 0
            for name, param in decoder_state_dict.items():
                if 'weight' in name:
                    mirror_name = str(4 - int(name[0])) + name[1:]  # Mirror symmetry
                    if mirror_name in encoder_state_dict:
                        decoder_param = encoder_state_dict[mirror_name].view(-1)
                        if i < decoder_param.size(0):
                            decoder_param_value = decoder_param[i].item()
                            break

            # Compute hash using HMAC with SHA-256
            message = str(int(decoder_param_value)  ^ i).encode()
            mac = hmac.new(self.secret_key.encode(), message, hashlib.sha256)
            position = int(mac.hexdigest(), 16) % total_params

            # Handle collisions with linear probing
            while bitmap[position]:
                position = (position + 1) % total_params

            # Embed the parameter
            param_vector[position] = encoder_vector[i]
            bitmap[position] = True

        # Reconstruct the model parameters
        start_idx = 0
        for name, info in layers_info.items():
            param = getattr(model_watermarked, name.split('.')[0]).weight
            for i in range(info['count']):
                param_size = torch.prod(torch.tensor(info['shape']))
                with torch.no_grad():
                    param[i] = param_vector[start_idx:start_idx + param_size].reshape(info['shape']).detach()
                start_idx += param_size

        return model_watermarked

    def extract(self, suspect_model, decoder_state_dict, config):
        """
        Extracts the embedded encoder from a suspect model

        Args:
            suspect_model: The model to extract watermark from
            decoder_state_dict: State dict of HufuNet's decoder (used for hashing)
            config: Configuration dict

        Returns:
            Extracted encoder parameters
        """
        # Extract all parameters from the model
        all_params = []

        for name, param in suspect_model.named_parameters():
            if 'conv' in name and 'weight' in name:
                for i in range(param.size(0)):
                    param_flat = param[i].view(-1)
                    all_params.append(param_flat)

        # Flatten all params
        param_vector = torch.cat(all_params)
        total_params = param_vector.size(0)

        # Prepare to reconstruct the encoder parameters
        encoder_state_dict = {}
        for name, param in decoder_state_dict.items():
            if 'weight' in name:
                # Create a mirror parameter name
                mirror_name = str(4 - int(name[0])) + name[1:]
                # Initialize with zeros of the same shape
                encoder_state_dict[mirror_name] = torch.zeros_like(param)

        # Compute total watermark size
        watermark_size = 0
        for name, param in encoder_state_dict.items():
            watermark_size += param.numel()

        # Extract each parameter using the hash function
        param_idx = 0
        for name, param in encoder_state_dict.items():
            param_flat = param.view(-1)
            for i in range(param_flat.size(0)):
                # Get the mirrored parameter from the decoder
                decoder_param_value = 0
                mirror_name = str(4 - int(name[0])) + name[1:]
                if mirror_name in decoder_state_dict:
                    decoder_param = decoder_state_dict[mirror_name].view(-1)
                    if i < decoder_param.size(0):
                        decoder_param_value = decoder_param[i].item()

                # Compute hash using HMAC with SHA-256
                message = str(decoder_param_value ^ param_idx).encode()
                mac = hmac.new(self.secret_key.encode(), message, hashlib.sha256)
                position = int(mac.hexdigest(), 16) % total_params

                # Handle collisions with linear probing
                original_position = position
                while position < total_params:
                    param_flat[i] = param_vector[position]
                    break
                    position = (position + 1) % total_params
                    if position == original_position:
                        break

                param_idx += 1

        return encoder_state_dict

    def verify(self, encoder_state_dict, decoder_state_dict, test_dataset, threshold=0.6):
        """
        Verifies if the extracted encoder combined with the decoder can reconstruct inputs

        Args:
            encoder_state_dict: Extracted encoder state dict
            decoder_state_dict: Owner's decoder state dict
            test_dataset: Dataset to test the reconstructed HufuNet
            threshold: MSE threshold for ownership verification

        Returns:
            Boolean indicating ownership and MSE loss
        """

        # Create a model with the extracted encoder and the owner's decoder
        class EncoderDecoder(nn.Module):
            def __init__(self, encoder_dict, decoder_dict):
                super(EncoderDecoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 20, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(20, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 7)
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 7),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 20, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(20, 1, 3, stride=2, padding=1, output_padding=1),
                    nn.Sigmoid()
                )

                # Load the state dictionaries
                self.encoder.load_state_dict(encoder_dict)
                self.decoder.load_state_dict(decoder_dict)

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded

        # Create a combined model
        combined_model = EncoderDecoder(encoder_state_dict, decoder_state_dict)

        # Evaluate the model on test dataset
        combined_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        combined_model = combined_model.to(device)

        criterion = nn.MSELoss()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, _ in test_dataset:
                data = data.to(device)
                _, reconstructed = combined_model(data)
                loss = criterion(reconstructed, data)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Verify ownership based on threshold
        return avg_loss < threshold, avg_loss