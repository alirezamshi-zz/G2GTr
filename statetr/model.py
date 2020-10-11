#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import logging
import math
import shutil
import tarfile
import tempfile
import sys
from io import open
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.cuda as cuda
from pytorch_pretrained_bert.modeling import load_tf_weights_in_bert,gelu,swish,\
    PRETRAINED_MODEL_ARCHIVE_MAP,BERT_CONFIG_NAME,TF_WEIGHTS_NAME,ACT2FN
from utils import relative_matmul, relative_matmul_dp, relative_matmul_dpv
from file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

logger = logging.getLogger(__name__)

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 num_labels,
                 label_emb,
                 POS_NULL,
                 graph_input = False,
                 num_hidden_layers=6,
                 num_attention_heads=12,
                 fcompmodel=True,
                 seppoint=0,
                 label_embedding=None,
                 layernorm=False,
                 topbuffer=False,
                 justexist=False,
                 hidden_size=768,
                 intermediate_size=3072,
                 dropout_char=0.33,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_labels = num_labels
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.fcompmodel = fcompmodel
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.label_emb = label_emb
            self.graph_input = graph_input
            self.POS_NULL = POS_NULL
            self.label_embedding = label_embedding
            self.layernorm_key = layernorm
            self.use_topbuffer = topbuffer
            self.use_justexist = justexist
            self.seppoint = seppoint
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class FFCompose(nn.Module):
    """
    This network piece takes the top two elements of the stack's embeddings
    and combines them to create a new embedding after an arc operation.

    The network architecture is:
    Inputs: 2 word embeddings (the head and the modifier embeddings)
    Output: Run through a linear layer -> tanh -> linear layer+skip connection
    """

    def __init__(self, embedding_dim,label_emb):

        super(FFCompose, self).__init__()

        self.emb_size = embedding_dim
        self.total_size = 2*embedding_dim + label_emb
        
        self.linear1 = nn.Linear(self.total_size,self.emb_size)
        nn.init.xavier_uniform_(self.linear1.weight)
        
        self.tanh = nn.Tanh()
        
        self.linear2 = nn.Linear(self.emb_size,self.emb_size)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, emb_head, emb_dep,label_emb):
        """
        First, concatenate emb_head,emb_dep, and label_emb into a single tensor.
        Then, apply linear -> tanh -> linear to the concatenated tensor to get a new representation.
        
        :param label_emb: The embedding of dependency label
        :param emb_dep:The embedding of the modifier in the arc operation
        :param emb_head: The embedding of the head in the arc operation
        :return The embedding of the combination as a row vector of shape (1, embedding_dim)
        """
        batch_size = len(emb_head)
        
        embeded = torch.cat((emb_head,emb_dep,label_emb),2).view(-1,self.total_size)
        out = self.linear1(embeded)
        out = self.tanh(out)
        out = self.linear2(out)
        return out.view(batch_size,-1,self.emb_size)
    
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.POS_NULL)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.graph_input:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size+1, config.hidden_size)
        else:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        ### composition model
        self.fcompmodel = config.fcompmodel
        self.graph_input = config.graph_input
        if config.fcompmodel:
            self.compose = FFCompose(config.hidden_size,config.label_emb)
        
        ### label embedding
        if config.fcompmodel or config.graph_input:
            self.label_emb = config.label_embedding

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, pos_ids, dep_ids, pos_dep_ids,
                label_dep, label_graph_dep, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        pos_embeddings_head = self.word_embeddings(pos_ids)
        words_embeddings_head = self.word_embeddings(input_ids)

        head_embeddings = words_embeddings_head + pos_embeddings_head
        
        if self.fcompmodel:
            label_embeddings = self.label_emb(label_dep)

        if self.graph_input:
            label_graph_embeddings = self.label_emb(label_graph_dep)
            
        if self.fcompmodel:
            pos_embeddings_dep = self.word_embeddings(pos_dep_ids)
            words_embeddings_dep = self.word_embeddings(dep_ids) 
            dep_embeddings = pos_embeddings_dep + words_embeddings_dep
            word_pos_embeddings = self.compose(head_embeddings,dep_embeddings,label_embeddings)
            word_pos_embeddings = word_pos_embeddings + head_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        if self.fcompmodel:
            if self.graph_input:
                embeddings = word_pos_embeddings + position_embeddings + \
                             token_type_embeddings + label_graph_embeddings
            else:
                embeddings = word_pos_embeddings + position_embeddings + token_type_embeddings
        else:
            if self.graph_input:
                embeddings = head_embeddings + position_embeddings +\
                             token_type_embeddings + label_graph_embeddings
            else:
                embeddings = head_embeddings + position_embeddings + token_type_embeddings
                
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        
        return embeddings

    
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        ### input graph mechanism
        self.graph_input = config.graph_input
        if self.graph_input:
            self.layernorm_key = config.layernorm_key
            self.num_dp = 3
            self.dp_relation_k = nn.Embedding(self.num_dp,self.attention_head_size,padding_idx=0)
            self.dp_relation_v = nn.Embedding(self.num_dp,self.attention_head_size,padding_idx=0)
            if self.layernorm_key:
                print("layernorm for keys")
                self.LayerNormKeys = torch.nn.LayerNorm(self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, graph_emb=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        ### add dependency relationships to attention weights
        if self.graph_input:
            dp_keys = self.dp_relation_k(graph_emb.to(key_layer.device))
            if self.layernorm_key:
                dp_keys = self.LayerNormKeys(dp_keys)

            dp_values = self.dp_relation_v(graph_emb.to(key_layer.device))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        query_key = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        query_key = query_key / math.sqrt(self.attention_head_size)

        attention_scores = query_key
        
        ## adding dependency graph to attention scores
        if self.graph_input: 
            attention_scores = attention_scores + relative_matmul_dp(query_layer, dp_keys)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)

        ## add dependency graph to value vectors
        if self.graph_input:
            context_layer = context_layer + relative_matmul_dpv(attention_probs, dp_values)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask,graph_emb=None):
        self_output = self.self(input_tensor, attention_mask, graph_emb)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and
                                                  isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, graph_emb=None):
        attention_output = self.attention(hidden_states, attention_mask, graph_emb)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask,graph_emb=None
                ,output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, graph_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.config = config
        if config.seppoint==0:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        elif config.use_topbuffer:
            if config.use_justexist:
                self.dense = nn.Linear(3 * config.hidden_size, config.hidden_size)
                self.dense_label = nn.Linear(2 * config.hidden_size, config.hidden_size)
            else:
                self.dense = nn.Linear(3 * config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, sep_point):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        if sep_point == 0:
            first_token_tensor = hidden_states[:, 0]
        else:
            if self.config.use_topbuffer:
                if self.config.use_justexist:
                    first_token_tensor = torch.cat((hidden_states[:,sep_point+2],hidden_states[:,sep_point],
                                            hidden_states[:,sep_point-1]),dim=1)
                    first_token_tensor_label = torch.cat((hidden_states[:,sep_point],hidden_states[:,sep_point-1]),dim=1)
                else:
                    first_token_tensor = torch.cat((hidden_states[:,sep_point+2],hidden_states[:,sep_point],
                                            hidden_states[:,sep_point-1]),dim=1)
            else:
                first_token_tensor = torch.cat((hidden_states[:,sep_point],hidden_states[:,sep_point-1]),dim=1)

        if self.config.use_justexist:
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)

            pooled_output_label = self.dense_label(first_token_tensor_label)
            pooled_output_label = self.activation(pooled_output_label)
            return [pooled_output,pooled_output_label]
        else:
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
            return pooled_output

## history model
class HistoryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,batch_first = True ,bias=True):
        super(HistoryLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,batch_first=batch_first,bias=bias)
        
    def forward(self,Input,prev):    
        Input = Input.unsqueeze(1)
        H0 = prev[0].contiguous()
        C0 = prev[1].contiguous()
        _,(actions_state,actions_cell) = self.rnn(Input,(H0,C0))
        
        return actions_state,actions_cell

class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

## Transformer encoder with/without G2G Tr
class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens pre-processing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. Its a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, sep_point, input_ids, pos_ids, dep_ids, pos_dep_ids, label_dep,label_graph,
                token_type_ids=None, attention_mask=None,graph_emb=None,
                output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        
        extended_attention_mask = (1.0 - extended_attention_mask) * (-10000.0)
        
        embedding_output = self.embeddings(input_ids, pos_ids, dep_ids,
                                           pos_dep_ids, label_dep,label_graph, token_type_ids)

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      graph_emb,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output,sep_point)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return pooled_output, encoded_layers
    

## classifier for predicting the label of dependency relation
class LabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_labels):
        super(LabelClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.activation = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size,n_labels)
        nn.init.xavier_uniform_(self.layer2.weight)
        
    def forward(self, input_label):
        output = self.layer1(input_label)
        output = self.activation(output)
        output = self.layer2(output)
        
        return output


## main parser model
class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and single hidden layer.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.
    """
    def __init__(self,
                 embeddings_shape,
                 device,
                 parser,
                 pad_action,
                 opt,
                 n_features=768):
        """ Initialize the parser model.

        @param embeddings (Tensor): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        
        ## initialization of parameters
        self.n_features = n_features
        self.n_classes = opt.nclass
        self.dropout_prob = opt.ffdropout
        self.hidden_size = opt.ffhidden
        self.embedding_size = embeddings_shape
        self.batch_size = opt.batchsize
        self.device = device
        self.n_layers_history = opt.nlayershistory
        self.max_step_length = opt.maxsteplength
        self.parser = parser
        self.pad_action = pad_action['P']
        self.num_labels = parser.n_transit-1
        self.hidden_size_label = opt.hiddensizelabel
        self.pooling_hid = opt.poolinghid
        self.fhistmodel = opt.fhistmodel
        self.use_justexist = opt.use_justexist
        ## initialization of embedding and bert model
        if opt.fcompmodel or opt.graphinput:
            self.label_emb = nn.Embedding(self.num_labels+1,self.n_features,padding_idx=self.num_labels)
        else:
            self.label_emb= None

        bertconfig = BertConfig(self.embedding_size, parser.n_transit-1,
                                              opt.labelemb, parser.P_NULL,opt.graphinput,
                                              opt.nattentionlayer,opt.nattentionheads,opt.fcompmodel,opt.seppoint,
                                              self.label_emb,opt.layernorm,opt.use_topbuffer,opt.use_justexist,
                                              opt.embsize,4*opt.embsize)
        self.bertmodel = BertModel(bertconfig)
        if opt.withbert:
            state_dict = torch.load('small_bert'+str(opt.outputname))
            self.bertmodel.load_state_dict(state_dict,strict=False)
            del state_dict
        else:
            state_dict_position = torch.load('position'+str(opt.outputname))
            self.bertmodel.embeddings.position_embeddings.load_state_dict(state_dict_position)
        
            if not opt.graphinput:
                state_dict_token = torch.load('token_type'+str(opt.outputname))
                self.bertmodel.embeddings.token_type_embeddings.load_state_dict(state_dict_token)
                del state_dict_token
        
            state_dict_word = torch.load('word_emb'+str(opt.outputname))
            self.bertmodel.embeddings.word_embeddings.load_state_dict(state_dict_word)

            del state_dict_position, state_dict_word

        ############################################################################################
        if opt.graphinput or opt.fcompmodel:
            self.bertmodel.embeddings.label_emb.weight[parser.n_transit-1].data.fill_(0.0)
            
        self.bertmodel.embeddings.word_embeddings.weight[parser.P_NULL].data.fill_(0.0)
        ############################################################################################
        
        ### initialization of lstm history model
        if self.fhistmodel:
            self.hist_size = opt.histsize
            self.action_emb = nn.Embedding(self.n_classes+1, self.hist_size)
            self.history = HistoryLSTM(self.hist_size,self.hist_size,self.n_layers_history)
        
            self.dtype = torch.cuda.FloatTensor if cuda.is_available() else torch.FloatTensor
            self.h0 = nn.Parameter(torch.rand(self.hist_size,requires_grad=True).type(self.dtype))
            self.c0 = nn.Parameter(torch.rand(self.hist_size,requires_grad=True).type(self.dtype))
            
        ## initialization of classifer
        if self.fhistmodel:
            self.embed_to_hidden = nn.Linear(self.n_features+self.hist_size, self.hidden_size)
        else:
            self.embed_to_hidden = nn.Linear(self.n_features, self.hidden_size)
            
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)
        
        ## initializtion of label-classifier
        if self.fhistmodel:
            self.label_classifier = LabelClassifier(self.n_features+self.hist_size,
                                                    self.hidden_size_label,self.num_labels)
        else:
            self.label_classifier = LabelClassifier(self.n_features, self.hidden_size_label,self.num_labels)
      
        

    def forward (self, mode_eval, sep_point,input_ids_x,pos_ids_x,batch_dep_input,batch_dep_pos_input,batch_dep_label_input,
                 batch_graph_label_input,attention_mask,update,token_type_ids,mask_stack,mask_buffer,index_stack,transitions=None,labels=None,
                 action_state=None,action_cell=None,graph_emb=None):
        """
        input:
        batch_partial_parses : a parser class which keep the state of stack,buffer and dependencies (now is list,
        change to torch) (batch_size,)
        
        batch_actions : (Tensor) gold actions of each instance (batch_size,seq_len)
        
        outputs: 
        
        outputs: (Tensor) output probabilities for each transition (seq_len, batch_size,len(actions))
        
        """
        batch_size = len(input_ids_x)
        #################################### history model ##################################
        ### initialization of history model
        if self.fhistmodel:
            if action_state is None:
                action_state = self.h0.unsqueeze(0).unsqueeze(1)\
                .expand(self.n_layers_history, batch_size, self.hist_size) # (batch_size, hid_dim, num_lstm_layers)
            if action_cell is None:
                action_cell = self.c0.unsqueeze(0).unsqueeze(1)\
                .expand(self.n_layers_history, batch_size, self.hist_size) # (batch_size, hid_dim, num_lstm_layers)
            
            
            #### update history model
            if transitions is not None:
                action_input = self.action_emb(transitions)
                label_input = self.label_emb(labels)
                ##### mean the label and action if not shift
                action_input = (label_input+action_input)/2.0 * (1-(transitions==2)).float().unsqueeze(1) + \
                                action_input * (transitions==2).float().unsqueeze(1)
       
                action_state, action_cell = self.history(action_input, (action_state, action_cell)) 
                # (batch_size, hid_dim, num_lstm_layers)
        ################################### bert model ########################################
            
        #### main model
        if self.pooling_hid:
            _,pooled_output = self.bertmodel(sep_point,input_ids_x, pos_ids_x,batch_dep_input, batch_dep_pos_input,
                                  batch_dep_label_input,batch_graph_label_input,token_type_ids, attention_mask,
                                             graph_emb,output_all_encoded_layers=False)
         
            pooled_output = torch.max(pooled_output,dim=1)[0]
        else:
            pooled_output,_ = self.bertmodel(sep_point,input_ids_x, pos_ids_x,batch_dep_input, batch_dep_pos_input,
                                  batch_dep_label_input,batch_graph_label_input,token_type_ids, attention_mask,graph_emb
                                             ,output_all_encoded_layers=False)

        ################################# classifer for transition ##############################
        if self.fhistmodel:
            if self.use_justexist:
                out_cat_org = torch.cat((pooled_output[0], action_state[-1, :, :]), 1)
                out_cat_org_label = torch.cat((pooled_output[1], action_state[-1, :, :]), 1)
            else:
                out_cat_org = torch.cat(( pooled_output, action_state[-1, :, :] ),1)
        else:
            if self.use_justexist:
                out_cat_org = pooled_output[0]
                out_cat_org_label = pooled_output[1]
            else:
                out_cat_org = pooled_output
        
        out_cat = self.relu(self.embed_to_hidden(out_cat_org))
        out_cat_drop = self.dropout(out_cat)
        outputs = self.hidden_to_logits(out_cat_drop)

        ################################## classifier for label ################################
        if self.use_justexist:
            outputs_label = self.label_classifier(out_cat_org_label)
        else:
            outputs_label = self.label_classifier(out_cat_org)

        ##################################### find next action ################################
        if mode_eval:
            mb_l = [self.parser.legal_labels( len(torch.nonzero(mask_stack[i])),
                                             len(torch.nonzero(mask_buffer[i])),
                                              index_stack[i]) for i in range(batch_size)]
                
            mb_l = torch.FloatTensor(mb_l).to(self.device)
            transitions = torch.max(torch.argmax(outputs + 10000 * mb_l, 1),update)
            
            labels = torch.max(outputs_label,1)[1]
            
            del mb_l

        if mode_eval:
            return transitions, labels, action_state,action_cell 
        else:
            return outputs, outputs_label,action_state,action_cell
        
        
        
        
        
        
        
        
        
      
