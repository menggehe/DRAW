""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch
from torch.nn import functional as F


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False, with_align=False, batch=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths, batch=batch)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)


class Discriminator(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Highway architecture based on the pooled feature maps is added. Dropout is adopted.
    """

    def __init__(self, num_classes, vocab_size, embedding_dim, filter_sizes, num_filters, dropout_prob):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f, (f_size, embedding_dim)) for f_size, num_f in zip(filter_sizes, num_filters)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(sum(num_filters), num_classes)

    def forward(self, x):
        """
        Inputs: x
            - x: (batch_size, seq_len)
        Outputs: out
            - out: (batch_size, num_classes)
        """
        emb = self.embed(x).unsqueeze(1)  # batch_size, 1 , seq_len , emb_dim

        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter *
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        out = torch.cat(pools, 1)  # batch_size * sum(num_filters)
        highway = self.highway(out)
        transform = F.sigmoid(highway)
        out = transform * F.relu(highway) + (1. - transform) * out  # sets C = 1 - T
        out = F.log_softmax(self.fc(self.dropout(out)), dim=1)  # batch * num_classes

        return out


class MATModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(MATModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # def forward(self, src, tgt, lengths,tgt_all,src_memory_bank,bptt=False, with_align=False, batch=None):
    def forward(self, pred, target, batch):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        # dec_in = tgt[:-1]  # exclude last target from inputs
        #
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, None)
        enc_state, memory_bank, lengths, output= self.encoder(src, src_lengths, batch=batch)

        # if bptt is False:
        #     self.decoder.init_state(src, memory_bank, enc_state)
        # loss = self.decoder(dec_in, memory_bank, tgt_all, output,src_memory_bank,
        #                     memory_lengths=lengths,
        #                     with_align=with_align)
        loss = self.decoder(pred, memory_bank)

        return loss


    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)