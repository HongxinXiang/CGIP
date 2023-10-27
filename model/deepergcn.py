import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from model.config.config_deepergcn import parse_args
from model.graph_sparse.torch_nn import norm_layer, MLP
from model.graph_sparse.torch_vertex import GENConv


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.set_output_type(output=0)  # default

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block
        self.conv_encode_edge = args.conv_encode_edge
        self.add_virtual_node = args.add_virtual_node

        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        norm = args.norm
        mlp_layers = args.mlp_layers

        graph_pooling = args.graph_pooling

        print('The number of layers {}'.format(self.num_layers),
              'Aggr aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))
        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        if self.add_virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden_channels)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(self.num_layers - 1):
                self.mlp_virtualnode_list.append(MLP([hidden_channels] * 3,
                                                     norm=norm))

        for layer in range(self.num_layers):
            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=self.conv_encode_edge, bond_encoder=True,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_channels)

        if not self.conv_encode_edge:
            self.bond_encoder = BondEncoder(emb_dim=hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception('Unknown Pool Type')

        self.graph_pred_linear = torch.nn.Linear(hidden_channels, 1)

    def set_output_type(self, output):
        self.output = output

    def forward(self, input_batch):
        # output_last: Whether to out the feature before classification layer
        x = input_batch.x
        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch

        h = self.atom_encoder(x)

        if self.add_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
            h = h + virtualnode_embedding[batch]

        if self.conv_encode_edge:
            edge_emb = edge_attr
        else:
            edge_emb = self.bond_encoder(edge_attr)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)

                if self.add_virtual_node:
                    virtualnode_embedding_temp = global_add_pool(h2, batch) + virtualnode_embedding
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer - 1](virtualnode_embedding_temp),
                        self.dropout, training=self.training)

                    h2 = h2 + virtualnode_embedding[batch]

                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        h_graph = self.pool(h, batch)

        if self.output == 0:
            return h_graph
        elif self.output == 1:
            return self.graph_pred_linear(h_graph)
        elif self.output == 2:
            return h_graph, self.graph_pred_linear(h_graph)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                print('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                print('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                print('Epoch {}, s {}'.format(epoch, ss))

    def global_max_mean_pool(self, x, batch):
        x_fp32 = x.float()
        x_max = global_max_pool(x_fp32, batch)
        x_mean = global_mean_pool(x_fp32, batch)
        x = torch.cat([x_max, x_mean], dim=-1).type_as(x)
        return x


def load_DeeperGCN(num_layers=3, dropout=0.5, block="res+", conv_encode_edge=False, add_virtual_node=False,
                   hidden_channels=256, num_tasks=1, conv="gen", gcn_aggr="max", t=1.0, learn_t=False, p=1.0,
                   learn_p=False, y=0.0, learn_y=False, msg_norm=False, learn_msg_scale=False, norm="batch",
                   mlp_layers=1, graph_pooling="mean"):

    assert block in ['res+', 'res', 'plain', 'dense']
    assert conv in ["gen"]
    assert gcn_aggr in ["mean", "max", "add", "softmax", "softmax_sg", "power"]
    assert norm in ["batch"]
    assert graph_pooling in ['mean', 'max', 'min']

    args = parse_args()
    args.num_layers = num_layers
    args.dropout = dropout
    args.block = block
    args.conv_encode_edge = conv_encode_edge
    args.add_virtual_node = add_virtual_node
    args.hidden_channels = hidden_channels
    args.num_tasks = num_tasks
    args.conv = conv
    args.gcn_aggr = gcn_aggr
    args.t = t
    args.learn_t = learn_t
    args.p = p
    args.learn_p = learn_p
    args.y = y
    args.learn_y = learn_y
    args.msg_norm = msg_norm
    args.learn_msg_scale = learn_msg_scale
    args.norm = norm
    args.mlp_layers = mlp_layers
    args.graph_pooling = graph_pooling

    return DeeperGCN(args)