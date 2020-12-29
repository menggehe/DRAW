"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, full_context_alignment=False,
                 alignment_heads=None):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads

    def forward(self, *args, **kwargs):
        """ Extend _forward for (possibly) multiple decoder pass:
        1. Always a default (future masked) decoder forward pass,
        2. Possibly a second future aware decoder pass for joint learn
            full context alignement.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, 1, model_dim)``
            * top_attn ``(batch_size, 1, src_len)``
            * attn_align ``(batch_size, 1, src_len)`` or None
        """
        with_align = kwargs.pop('with_align', False)
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads is not None:
                attns = attns[:, :self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, attn_align

    def _forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                 layer_cache=None, step=None, future=False):
        """ A naive forward pass for transformer decoder.
        # TODO: change 1 to T as T could be 1 or tgt_len
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attns ``(batch_size, head, 1, src_len)``

        """
        dec_mask = None

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,
                                      attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, _ = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(memory_bank, memory_bank, query_norm,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout
class TransformerDecoderLayer_match(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 aan_useffn=False, full_context_alignment=False,
                 alignment_heads=None):
        super(TransformerDecoderLayer_match, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model,
                                              dropout=attention_dropout,
                                              aan_useffn=aan_useffn)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads

    def forward(self, *args, **kwargs):
        """ Extend _forward for (possibly) multiple decoder pass:
        1. Always a default (future masked) decoder forward pass,
        2. Possibly a second future aware decoder pass for joint learn
            full context alignement.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, 1, model_dim)``
            * top_attn ``(batch_size, 1, src_len)``
            * attn_align ``(batch_size, 1, src_len)`` or None
        """
        with_align = kwargs.pop('with_align', False)
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        if with_align:
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads is not None:
                attns = attns[:, :self.alignment_heads, :, :].contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, attn_align

    def _forward(self, inputs, tgt_pad_mask,
                 layer_cache=None, step=None, future=False):
        """ A naive forward pass for transformer decoder.
        # TODO: change 1 to T as T could be 1 or tgt_len
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attns ``(batch_size, head, 1, src_len)``

        """
        dec_mask = None

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, _ = self.self_attn(input_norm, input_norm, input_norm,
                                      mask=dec_mask,
                                      layer_cache=layer_cache,
                                      attn_type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, _ = self.self_attn(input_norm, mask=dec_mask,
                                      layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs
        output = self.feed_forward(query)

        return output, _

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads=None):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
             attention_dropout, self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             aan_useffn=aan_useffn,
             full_context_alignment=full_context_alignment,
             alignment_heads=alignment_heads)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop('with_align', False)
        attn_aligns = []

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                with_align=with_align)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
def func_attention(query, contextT, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    sourceL = contextT.size(1)

    # --> batch x sourceL x ndf
    # context = context.view(batch_size, -1, sourceL)
    # contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(contextT.transpose(1,2), attnT)

    return weightedContext, attn.view(batch_size, -1, sourceL)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

class TransformerDecoder_match(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads=None):
        super(TransformerDecoder_match, self).__init__()

        self.embeddings = embeddings
        # Decoder State
        self.state = {}

        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer_match(d_model, heads, d_ff, dropout,
                                     attention_dropout, self_attn_type=self_attn_type,
                                     max_relative_positions=max_relative_positions,
                                     aan_useffn=aan_useffn,
                                     full_context_alignment=full_context_alignment,
                                     alignment_heads=alignment_heads)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.alignment_layer = alignment_layer
        self.lstm = nn.LSTMCell(512, 512)
        self.FC = nn.Linear(512, 1)
        self.FC2 = nn.Linear(512, 512)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
                                        is list else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.aan_useffn,
            opt.full_context_alignment,
            opt.alignment_layer,
            alignment_heads=opt.alignment_heads)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def init_hidden(self, batch_size):
        # h = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        # c = Variable(torch.zeros((1, batch_size, self.hidden_dim)))
        h = Variable(torch.zeros((batch_size, 512)))
        c = Variable(torch.zeros((batch_size, 512)))
        h, c = h.cuda(), c.cuda()
        return h, c

    # def forward(self, tgt, memory_bank, tgt_all, global_graph, src_memory_bank,step=None, eps=1e-8, **kwargs):
    def forward(self, tgt_all,  src_memory_bank,step=None,**kwargs):
        """Decode, possibly stepwise."""
        # if step == 0:
        #     self._init_cache(memory_bank)
        tgt_all=tgt_all.transpose(0,1).unsqueeze(2)
        emb = self.embeddings(tgt_all, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim
        tgt_words = tgt_all[:, :, 0].transpose(0, 1)

        output = emb.transpose(0, 1).contiguous()
        h0, c0 = self.init_hidden(emb.size(1))
        outputs=[]
        for i, k in enumerate(emb):
            h0, c0 = self.lstm(k, (h0, c0))
            outputs.append(h0)
        output = torch.stack(outputs, 1)
        # global_text=c0
        # pad_idx = self.embeddings.word_padding_idx
        # tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
        # with_align = kwargs.pop('with_align', False)
        # attn_aligns = []
        # for i, layer in enumerate(self.transformer_layers):
        #     layer_cache = self.state["cache"]["layer_{}".format(i)] \
        #         if step is not None else None
        #     output, attn, attn_align = layer(
        #         output,
        #         tgt_pad_mask,
        #         layer_cache=layer_cache,
        #         step=step,
        #         with_align=with_align)
        #     if attn_align is not None:
        #         attn_aligns.append(attn_align)
        #
        # output = self.layer_norm(output).contiguous()
        # attn_weight = F.softmax(self.FC(output), dim=1)
        # global_text = attn_weight * output
        # global_text = self.FC2(global_text.sum(dim=1))
        # # --> seq_len x batch_size x nef
        # if global_text.dim() == 2:
        #     global_text = global_text.unsqueeze(0).data.float()
        #     global_graph = global_graph.unsqueeze(0).data.float()
        #
        # # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
        # global_graph_norm = torch.norm(global_graph, 2, dim=2, keepdim=True).data.float()
        # global_text_norm = torch.norm(global_text, 2, dim=2, keepdim=True).data.float()
        # # scores* / norm*: seq_len x batch_size x batch_size
        # scores0 = torch.bmm(global_graph, global_text.transpose(1, 2))
        # norm0 = torch.bmm(global_graph_norm, global_text_norm.transpose(1, 2))
        # scores0 = scores0 / norm0.clamp(min=eps) * 10.0
        # # --> batch_size x batch_size
        # scores0 = torch.squeeze(scores0, 0)
        # # print(scores0.dim())
        # scores1 = scores0.transpose(0, 1)

        target = src_memory_bank.transpose(0, 1).contiguous()
        similarities = []
        for i, k in enumerate(output):
            k = k.unsqueeze(0).transpose(1, 2)
            k = k.repeat(output.size(0), 1, 1)
            """
                        weiContext: batch x nef x words_num
                        attn: batch x words_num x 24
                    """
            weiContext, attn = func_attention(k, target, 5.0)
            # --> batch_size x words_num x nef
            k = k.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()
            # --> batch_size*words_num x nef
            k = k.view(output.size(0) * output.size(1), -1)
            weiContext = weiContext.view(output.size(0) * output.size(1), -1)
            #
            # -->batch_size*words_num
            row_sim = cosine_similarity(k, weiContext)
            # --> batch_size x words_num
            row_sim = row_sim.view(output.size(0), output.size(1))

            # Eq. (10)
            row_sim.mul_(5.0).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)

            # --> 1 x batch_size
            # similarities(i, j): the similarity between the i-th image and the j-th text description
            similarities.append(row_sim)
        # batch_size x batch_size
        similarities = torch.cat(similarities, 1)
        similarities = similarities * 10.0
        similarities1 = similarities.transpose(0, 1)
        labels = Variable(torch.LongTensor(range(output.size(0)))).detach().cuda()
        # label =torch.eye(output.size(0)).detach().cuda()

        pred = torch.argmax(similarities, dim=1)
        pred_list = pred.cpu().numpy().tolist()
        acc = 0
        for i in range(len(pred_list)):
            if pred_list[i] == i:
                acc = acc + 1
        acc = acc / (i + 1)
        # loss2 = nn.CrossEntropyLoss()(scores0, labels)
        # loss3 = nn.CrossEntropyLoss()(scores1, labels)
        # loss0 = nn.BCELoss()(similarities, label)
        reward=F.log_softmax(similarities/10)
        # loss0 = nn.CrossEntropyLoss()(similarities, labels)
        # loss1 = nn.CrossEntropyLoss()(similarities1, labels)

        # return loss0 + loss1 + loss2 + loss3
        return reward


        # pad_idx = self.embeddings.word_padding_idx
        # src_lens = kwargs["memory_lengths"]
        # src_max_len = self.state["src"].shape[0]
        # src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        # tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
        #
        # with_align = kwargs.pop('with_align', False)
        # attn_aligns = []
        #
        # for i, layer in enumerate(self.transformer_layers):
        #     layer_cache = self.state["cache"]["layer_{}".format(i)] \
        #         if step is not None else None
        #     output, attn, attn_align = layer(
        #         output,
        #         src_memory_bank,
        #         src_pad_mask,
        #         tgt_pad_mask,
        #         layer_cache=layer_cache,
        #         step=step,
        #         with_align=with_align)
        #     if attn_align is not None:
        #         attn_aligns.append(attn_align)
        #
        # output = self.layer_norm(output)
        # dec_outs = output.transpose(0, 1).contiguous()
        # attn = attn.transpose(0, 1).contiguous()
        #
        # attns = {"std": attn}
        # if self._copy:
        #     attns["copy"] = attn
        # if with_align:
        #     attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
        #     # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg
        #
        # # TODO change the way attns is returned dict => list or tuple (onnx)
        # return dec_outs, attns,src_memory_bank

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth),
                                                    device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)
