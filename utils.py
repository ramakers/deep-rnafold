import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import make_grid

NUM_ATOMS = 8
max_len = 100


def get_3D_tensor(array_a, *args):
    """
    input(s):
        either
            coords [Ra*8, 3] and optionally [Rb*8, 3]
        or
            mask [Ra*8] and optionally [Rb*8]

    output:
        either
            D [Ra, Ra, 64]
        or
            D [Ra, Rb, 64]
    """
    if len(args) == 1:
        array_b = args[0]
    else:
        array_b = array_a

    array_a_shape = array_a.shape
    array_b_shape = array_b.shape
    if array_a_shape[0] % NUM_ATOMS != 0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_a_shape)
    if array_b_shape[0] % NUM_ATOMS != 0:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_b_shape)

    if len(array_a_shape) != len(array_b_shape):
        raise Exception('Inputs have different dimensions:', array_a_shape, array_b_shape)

    # assume that input was "coords"
    if len(array_a_shape) == 2:
        Ra = len(array_a) // NUM_ATOMS
        Rb = len(array_b) // NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1, 3])  # [R*8, 3] -> [R, 1, 8, 1, 3]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS, 3])  # [R*8, 3] -> [1, R, 1, 8, 3]

        D = np.sqrt(np.sum((A - B) ** 2, axis=4, keepdims=True))  # [R, R, 8, 8, 1]
        D = np.reshape(D, [Ra, Rb, NUM_ATOMS ** 2])  # [R, R, 8, 8, 1] -> [R, R, 64]

    # assume that input was "mask"
    elif len(array_a_shape) == 1:
        Ra = len(array_a) // NUM_ATOMS
        Rb = len(array_b) // NUM_ATOMS
        A = array_a.reshape([Ra, 1, NUM_ATOMS, 1])  # [R*8] -> [R, 1, 8, 1]
        B = array_b.reshape([1, Rb, 1, NUM_ATOMS])  # [R*8] -> [1, R, 1, 8]
        D = A * B
        D = np.reshape(D, [Ra, Rb, NUM_ATOMS ** 2])

    else:
        raise Exception('Shape must be [R*8, 3] or [R*8], but is', array_a_shape)

    # symmetrize
    if Ra == Rb:
        triu = np.reshape(np.triu(np.ones([Ra, Ra]), 1), [Ra, Ra, 1])
        triu_plus_diag = np.reshape(np.triu(np.ones([Ra, Ra]), 0), [Ra, Ra, 1])
        D = D * triu_plus_diag + np.transpose(D * triu, axes=[1, 0, 2])

    return D


class MaskedEncodedRNA(Dataset):
    def __init__(self, filename, min_num_mask=0, max_num_mask=0):
        super().__init__()
        self.min_num_mask = min_num_mask
        self.max_num_mask = max_num_mask
        self.nuc2ind_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 4}

        with open(filename, 'rb') as file:
            self.examples = pickle.load(file)
        self.keys = list(self.examples.keys())

    def __getitem__(self, idx):
        key = self.keys[idx]
        z, raw_coords, raw_loss_mask, nuc, seq_len = self.examples[key]
        z = torch.from_numpy(z).long()

        mask = torch.ones(max_len * max_len).long()
        mask_number = np.random.randint(self.min_num_mask, self.max_num_mask)
        mask_indices = np.random.choice(list(range(max_len * max_len)), mask_number, replace=False)
        mask[mask_indices] = 0

        z_masked = (z + 1) * mask.view(max_len, max_len)
        z_masked = z_masked.long()

        dist = torch.from_numpy(get_3D_tensor(raw_coords)).permute(2, 0, 1)
        dist = F.pad(dist, [0, max_len - seq_len, 0, max_len - seq_len]).float()

        loss_mask = torch.from_numpy(get_3D_tensor(raw_loss_mask)).permute(2, 0, 1)
        loss_mask = F.pad(loss_mask, [0, max_len - seq_len, 0, max_len - seq_len]).long()

        nuc_idx = torch.tensor(list(map(lambda c: self.nuc2ind_dict[c], nuc)))
        nuc_enc = F.one_hot(nuc_idx, 5).view(-1)
        nuc_enc = F.pad(nuc_enc, [0, (max_len-seq_len)*5]).long()

        return z, z_masked, nuc, nuc_enc, dist, loss_mask, seq_len, key

    def __len__(self):
        return len(self.keys)


def get_pad_mask(seq_len, embed_dim, device=torch.device('cpu')):
    bs = len(seq_len)
    pad_mask = torch.zeros(bs, embed_dim, max_len, max_len).long().to(device)
    for i, s in enumerate(seq_len):
        pad_mask[i, :, :s, :s] = 1
    return pad_mask


def print_data(dataset, code_book, vqvae, device=torch.device('cpu')):
    num_embed, embed_dim = code_book.size()
    z, _, nuc, _, dist, loss_mask, seq_len, key = dataset[41]
    z = z.to(device)
    dist = dist * loss_mask

    z_q = F.embedding(z.unsqueeze(0), code_book).permute(0, 3, 1, 2)
    pad_mask = get_pad_mask([seq_len], embed_dim, device=device)
    recon = vqvae.decode(z_q * pad_mask).squeeze().detach().cpu()
    recon = recon * loss_mask

    z = F.one_hot(z, num_embed).permute(2, 0, 1).cpu()

    plt.figure(figsize=(25, 5))
    plt.suptitle(f'Quantized Latent Space and Distance Matrix\n'
                 f'Key: {key}\n'
                 f'Nuc: {nuc}', fontsize=14)

    for i in range(num_embed):
        plt.subplot(1, 7, 1+i)
        plt.title(f'quant dim {i}')
        plt.imshow(z[i, :seq_len + 5, :seq_len + 5])
        plt.axis('off')

    plt.subplot(1, 7, 4)
    plt.title('orig ch 0')
    plt.imshow(dist[0, :seq_len+5, :seq_len+5])
    plt.axis('off')

    plt.subplot(1, 7, 5)
    plt.title('recon ch 0')
    plt.imshow(recon[0, :seq_len + 5, :seq_len + 5])
    plt.axis('off')

    plt.subplot(1, 7, 6)
    plt.title('orig ch 1')
    plt.imshow(dist[1, :seq_len+5, :seq_len+5])
    plt.axis('off')

    plt.subplot(1, 7, 7)
    plt.title('recon ch 1')
    plt.imshow(recon[1, :seq_len+5, :seq_len+5])
    plt.axis('off')

    plt.show()


def quadratic_discriminator(logits, code_book, embedding_variance):
    """
    params: logits [bs, embed_dim, w, h]
    """
    bs, c, w, h = logits.size()
    num_embed, embed_dim = code_book.size()
    assert c == embed_dim
    logits_p = logits.permute(0, 2, 3, 1).reshape(bs * w * h, c).repeat_interleave(num_embed, dim=0)
    logits_p = logits_p.reshape(bs * w * h, num_embed, c)  # [bs * L * L, num_embed, embed_dim]
    diff = code_book.unsqueeze(0) - logits_p  # [bs * L * L, num_embed, embed_dim]
    diff = diff.unsqueeze(-1)  # [bs * L * L, num_embed, embed_dim, 1]
    diff_t = diff.transpose(dim0=2, dim1=3)  # [bs * w * h, num_embed, 1, embed_dim]
    l2 = (diff_t @ diff).squeeze()  # [bs * L * L, num_embed]
    l2_norm = -0.5 * l2 / embedding_variance.unsqueeze(0)  # [bs * L * L, num_embed]
    return l2_norm.reshape(bs, w, h, num_embed).permute(0, 3, 1, 2)  # [bs, num_embed, L, L]


def train_plots(model, z, z_logits, key, seq_len, num_embed, epoch=None):
    """blub"""
    model.eval()
    with torch.no_grad():
        z = F.one_hot(z, num_embed).permute(0, 3, 1, 2)

        plt.figure(figsize=(25, 8))
        plt.suptitle(f"Quantized latent space\n"
                     f"Keys: {key[0]} and {key[1]}")

        k = 0
        for i in range(2):
            n = seq_len[i]
            for j in range(3):
                k += 1
                plt.subplot(2, 6, k)
                plt.title(f'Target z{i}_{j}')
                plt.imshow(z[i, j, :n + 5, :n + 5].cpu())
                plt.axis('off')

        for i in range(2):
            n = seq_len[i]
            for j in range(3):
                k += 1
                plt.subplot(2, 6, k)
                plt.title(f'Model z{i}_{j} (Logits)')
                plt.imshow(z_logits[i, j, :n + 5, :n + 5].detach().cpu())
                plt.axis('off')

        if epoch is not None:
            plt.savefig(f'./outputs/rna_training_epoch_{epoch+1}.png')
        plt.show()
