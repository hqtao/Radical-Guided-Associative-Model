# coding: utf-8
# 2020/1/2 @ tongshiwei
import mxnet as mx
from longling.ML.MxnetHelper.gallery.layer.attention import DotProductAttentionCell
from mxnet import gluon
from .net import EmbeddingLSTM

from CangJie.SE.bert import CharBert


class WCLSTM(EmbeddingLSTM):
    def __init__(self, net_type,
                 class_num, embedding_dim,
                 lstm_hidden,
                 embed_dropout=0.5, fc_dropout=0.5, embedding_size=None,
                 associate_word=True, enable_attention=True, average=False,
                 **kwargs):
        r"""Baseline: 仅包含词和字，不包括部首的网络模型"""
        super(WCLSTM, self).__init__(**kwargs)
        self.word_length = None
        self.lstm_hidden = lstm_hidden if lstm_hidden is not None else embedding_dim
        self.net_type = net_type
        self.associate_word = associate_word
        self.enable_attention = enable_attention
        self.average = average

        with self.name_scope():
            self.embedding = AWEmbedding(
                word_embedding_size=embedding_size["w"],
                embedding_dim=embedding_dim,
                dropout=embed_dropout,
            )
            if not self.associate_word:
                self.enable_attention = False
            if self.net_type == "lstm":
                setattr(
                    self, "rnn",
                    gluon.rnn.LSTMCell(self.lstm_hidden)
                )
            elif self.net_type == "bilstm":
                setattr(
                    self, "rnn",
                    gluon.rnn.BidirectionalCell(
                        gluon.rnn.LSTMCell(self.lstm_hidden),
                        gluon.rnn.LSTMCell(self.lstm_hidden)
                    )
                )
            else:
                raise TypeError(
                    "net_type should be either lstm or bilstm, now is %s"
                    % self.net_type
                )

            if self.enable_attention:
                self.bert_attention = DotProductAttentionCell(
                    units=self.lstm_hidden, scaled=False
                )
                self.lstm_attention = DotProductAttentionCell(
                    units=self.lstm_hidden, scaled=False
                )
            self.fc = gluon.nn.HybridSequential()
            self.fc.add(
                gluon.nn.Activation("relu"),
                # gluon.nn.ELU(),
                # gluon.nn.GELU(),
                # gluon.nn.Dropout(fc_dropout),
                # gluon.nn.Dense(self.lstm_hidden, activation="relu"),
                gluon.nn.Dropout(fc_dropout),
                gluon.nn.Dense(class_num)
            )

    def hybrid_forward(self, F, seq, cls, aw, word_mask, associate_word_mask, *args, **kwargs):
        seq_length = self.word_length if self.word_length else len(seq[0])

        if self.associate_word:
            associate_word_embedding = self.embedding(aw)

        merge_outputs = True
        if self.net_type == "lstm":
            w_e, (w_o, w_s) = getattr(self, "rnn").unroll(
                seq_length,
                seq,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )

        elif self.net_type == "bilstm":
            w_e, (w_lo, w_ls, w_ro, w_rs) = getattr(self, "rnn").unroll(
                seq_length, seq, merge_outputs=merge_outputs,
                valid_length=word_mask)

            w_o = F.concat(w_lo, w_ro)
        else:
            raise TypeError(
                "net_type should be either lstm or bilstm, now is %s"
                % self.net_type
            )

        if self.associate_word:
            if self.enable_attention:
                caw_le = self.lstm_attention(
                    F.expand_dims(w_o, axis=1), associate_word_embedding, mask=associate_word_mask
                )[0]
                caw_be = self.bert_attention(
                    F.expand_dims(cls, axis=1), associate_word_embedding, mask=associate_word_mask
                )[0]

                caw_lo = F.sum(caw_le, axis=1)
                caw_bo = F.sum(caw_be, axis=1)
                if self.average:
                    caw_lo = F.broadcast_div(caw_lo, F.expand_dims(associate_word_mask, axis=1))
                    caw_bo = F.broadcast_div(caw_bo, F.expand_dims(associate_word_mask, axis=1))

                fc_in = F.concat(w_o, cls, caw_lo, caw_bo)
            else:
                awe = F.sum(associate_word_embedding, axix=1)
                if self.average:
                    awe = F.broadcast_div(
                        awe, F.expand_dims(associate_word_mask, axis=1)
                    )
                fc_in = F.concat(w_o, cls, awe)
        else:
            fc_in = F.concat(w_o, cls)

        return self.fc(fc_in)

    def set_network_unroll(self, word_length, character_length):
        self.word_length = word_length


class AWEmbedding(gluon.HybridBlock):
    def __init__(self, word_embedding_size,
                 embedding_dim, dropout=0.5, prefix=None,
                 params=None):
        super(AWEmbedding, self).__init__(prefix, params)
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(
                word_embedding_size, embedding_dim
            )
            self.word_dropout = gluon.nn.Dropout(dropout)

    def hybrid_forward(self, F, aw_seq, *args, **kwargs):
        word_embedding = self.word_embedding(aw_seq)

        associate_word_embedding = self.word_dropout(word_embedding)

        return associate_word_embedding

    def set_weight(self, embeddings):
        self.word_embedding.weight.set_data(embeddings["w"])
