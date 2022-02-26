import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv
import dgl.function as fn
import dgl.ops as ops
import scipy.sparse


class GDNLayer(nn.Module):
    '''
    A graph differentiate layer, proposed by
    *Inductive Anomaly Detection on Attributed Networks* (IJCAI'20)
    z_i = W_1 \cdot h_i                         (Eq.1)
    delta_{ij} =  h_i - h_j                     (Eq.2)
    d'_{ij} = W_2 \cdot delta_{ij}              (Eq.3)
    e_{ij} = a \cdot d'_{ij}                    (Eq.4)
    alpha = softmax(e)                          (Eq.5)
    h_next = \sigma(z + \sum alpha * diff_prime)(Eq.6)
    '''
    def __init__(self, in_dim, out_dim, activation=F.elu) -> None:
        '''init a gdn layer'''
        super(GDNLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)        # Eq.1
        self.diff_fc = nn.Linear(in_dim, out_dim, bias=False)   # Eq.3
        self.att = nn.Parameter(torch.FloatTensor(size=(out_dim, 1)))
        self.activation = activation
        self.reset_parameters()
    
    def reset_parameters(self):
        '''(re)init parameters'''
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.diff_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.att, gain=gain)

    def message_func(self, edges):
        '''message used to aggregate neighbor information'''
        return {'m': edges.data['a'] * edges.data['d']}

    def reduce_func(self, nodes):
        '''aggregate messages and output'''
        # W_1 \cdot h_i + \sum a_{ij} \cdot 
        h = nodes.data['z'] + torch.sum(nodes.mailbox['m'], dim=1)   # Eq.6
        if self.activation:
            h = self.activation(h)
        return {'h': h}

    def forward(self, graph, h):
        '''forward pass'''
        with graph.local_scope():   # avoid changing original graph
            z = self.fc(h)                                      # Eq.1
            graph.ndata['z'] = z
            graph.ndata['h_d'] = self.diff_fc(h)                # Eq.3
            graph.apply_edges(fn.u_sub_v('h_d', 'h_d', 'd'))    # Eq.2
            e =  F.leaky_relu((graph.edata['d'] @ self.att).sum(dim=-1, keepdim=True))   # Eq.4
            graph.edata['a'] = ops.edge_softmax(graph, e)
            graph.update_all(self.message_func, self.reduce_func)
            h_next = graph.ndata.pop('h')
            return h_next


class GDN_Autoencoder(nn.Module):
    '''
    Graph autoencoder using the Graph Differentiate Layer
    h_i^{(l, k)} = GDN_l(A^k, X)                (Eq.1)
    e_i^{(l, k)} = a \cdot h_i^{(l, k)}         (Eq.2)
    a = softmax(e)                              (Eq.3)
    h_i^l = \sum_{k=1}^K beta * h_i^{(l, k)}    (Eq.4)
    '''
    def __init__(self, in_dim, hidden_dim, layer='gdn', k=3) -> None:
        super(GDN_Autoencoder, self).__init__()
        if layer == 'gdn':
            self.encoder = GDNLayer(in_dim, hidden_dim, activation=F.elu)
            self.decoder = GDNLayer(hidden_dim, in_dim, activation=None)
        elif layer == 'gcn':
            self.encoder = dglnn.GraphConv(in_dim, hidden_dim, activation=F.elu)
            self.decoder = dglnn.GraphConv(hidden_dim, in_dim, activation=None)
        self.att = nn.Parameter(torch.FloatTensor(size=(hidden_dim, 1)))
        nn.init.xavier_normal_(self.att, gain=nn.init.calculate_gain('relu'))
        self.k = k

    def forward(self, graph, h):
        '''forward pass'''
        adj_mat = graph.adjacency_matrix().to_dense().numpy()
        As = [adj_mat]
        for _ in range(self.k - 1):
            As.append(As[-1] @ adj_mat) 
        
        Zs = []
        device = graph.device
        with graph.local_scope():
            # encoder
            for A in As:                                        # Eq.1
                graph = dgl.from_scipy(scipy.sparse.coo_matrix(A)).to(device)
                Zs.append(self.encoder(graph, h))
            e = [Z @ self.att for Z in Zs]                      # Eq.2
            a = F.softmax(F.leaky_relu(torch.cat(e, dim=-1)), dim=-1) # Eq.3
            z = torch.sum(a.view(-1, 1, self.k) * torch.stack(Zs, dim=-1), dim=-1) # Eq.4
            # decoder
            x = self.decoder(graph, z)
            return x


class Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim) -> None:
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input)


class Reconstruct(nn.Module):
    '''reconstruct the adjacent matrix and rank anomalies'''
    def __init__(self, **kwargs):
        super(Reconstruct, self).__init__(**kwargs)
    
    def forward(self, h):
        return torch.mm(h, h.transpose(1, 0))


class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout=0.5):
        super(Dominant, self).__init__()
        self.shared_encoder = nn.ModuleList(
            [GraphConv(
                in_feats=feat_size,
                out_feats=hidden_size,
                activation=F.relu,
                bias=False
            ),
            GraphConv(
                in_feats=hidden_size,
                out_feats=hidden_size,
                activation=F.relu,
                bias=False
            ),
            GraphConv(
                in_feats=hidden_size,
                out_feats=hidden_size,
                activation=F.relu,
                bias=False
            )]
        )
        self.attr_decoder = GraphConv(
            in_feats=hidden_size,
            out_feats=feat_size,
            activation=F.relu,
            bias=False
        )
        self.struct_decoder = nn.Sequential(
            Reconstruct(),
            nn.Sigmoid()
        )

    def embed(self, g, h):
        for layer in self.shared_encoder:
            h = layer(g, h)
        return h
    
    def forward(self, g, h):
        # encode
        h = self.embed(g, h)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(h)
        # decode feature matrix
        x_hat = self.attr_decoder(g, h)
        # return reconstructed matrices
        return struct_reconstructed, x_hat


class NodeClassification(nn.Module):
    def __init__(self, feat_size, hidden_size, num_class, activation=F.relu):
        super(NodeClassification, self).__init__()
        self.nn = nn.ModuleList(
            [GraphConv(
                in_feats=feat_size,
                out_feats=hidden_size,
                activation=activation,
                bias=False
            ),
            GraphConv(
                in_feats=hidden_size,
                out_feats=hidden_size,
                activation=activation,
                bias=False
            ),
            GraphConv(
                in_feats=hidden_size,
                out_feats=num_class,
                activation=activation,
                bias=False
            )]
        )

    def forward(self, g, h):
        for layer in self.nn:
            h = layer(g, h)
        return h


class _Discriminator(nn.Module):
    def __init__(self, in_feat1, in_feat2):
        super(_Discriminator, self).__init__()
        self.disc = nn.Bilinear(in_feat1, in_feat2, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h_attr, h_attr_shfs, h_graph, h_graph_shfs):
        logit1 = self.disc(h_attr, h_graph)
        logit2 = self.disc(h_attr_shfs, h_graph)
        logit3 = self.disc(h_attr, h_graph_shfs)
        logit4 = self.disc(h_attr_shfs, h_graph_shfs)
        return torch.cat([logit1, logit2, logit3, logit4], axis=-1)        


class ContrastAD(nn.Module):
    '''
    the model for contrastive anomlay detection
    '''
    def __init__(self, graph_feat_size, attr_feat_size, graph_hidden_size, attr_hidden_size):
        '''
        params:
            graph_feat_size: structural feature size
            attr_feat_size: attribute size
            graph_hidden_size: gnn hidden size
            attr_hidden_size: attribute mlp hidden size
        '''
        super(ContrastAD, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=attr_feat_size,
                out_features=attr_hidden_size    
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=attr_hidden_size,
                out_features=attr_hidden_size
            ),
            nn.ReLU()
        )
        
        self.graph_encoder_1 = GraphConv(
            in_feats=graph_feat_size,
            out_feats=graph_hidden_size,
            activation=nn.ReLU()
        )
        
        self.graph_encoder_2 = GraphConv(
            in_feats=graph_hidden_size,
            out_feats=graph_hidden_size,
            activation=nn.ReLU()
        )
        
        self.disc = _Discriminator(attr_hidden_size, graph_hidden_size)

    def forward(self, g, graph_feat, graph_feat_shfs, attr_feat, attr_feat_shfs):
        '''
        forward pass
        '''
        # positive, attributes
        h_feat = self.mlp(attr_feat)
        # negative, attributes
        h_feat_shfs = self.mlp(attr_feat_shfs)
        # positive, structural
        h_graph = self.graph_encoder_1(g, graph_feat)
        h_graph = self.graph_encoder_2(g, h_graph)
        # negative, structural
        h_graph_shfs = self.graph_encoder_1(g, graph_feat_shfs)
        h_graph_shfs = self.graph_encoder_2(g, h_graph_shfs)

        return self.disc(h_feat, h_feat_shfs, h_graph, h_graph_shfs)

    def score(self, g, graph_feat, attr_feat):
        '''return discrimination score'''
        h_attr = self.mlp(attr_feat)
        h_graph = self.graph_encoder_1(g, graph_feat)
        h_graph = self.graph_encoder_2(g, h_graph)

        return self.disc.disc(h_attr, h_graph)


class GRL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_num=1, head_num=1) -> None:
        super(GRL, self).__init__()
        self.shared_encoder = nn.ModuleList(
            GraphConv(
                in_dim if i==0 else hidden_dim,
                (out_dim if i == hidden_num - 1 else hidden_dim),
                activation=torch.sigmoid
            )
            for i in range(hidden_num)
        )
        self.attr_decoder = GraphConv(
            in_feats=out_dim,
            out_feats=in_dim,
            activation=torch.sigmoid,
        )
        self.struct_decoder = nn.Sequential(
            Reconstruct(),
            nn.Sigmoid()
        )
        self.dense = nn.Sequential(nn.Linear(out_dim, out_dim))

    def embed(self, g, h):
        for layer in self.shared_encoder:
            h = layer(g, h).view(h.shape[0], -1)
        # h = self.project(g, h).view(h.shape[0], -1)
        # return h.div(torch.norm(h, p=2, dim=1, keepdim=True))
        return self.dense(h)
    
    def reconstruct(self, g, h):
        struct_reconstructed = self.struct_decoder(h)
        x_hat = self.attr_decoder(g, h).view(h.shape[0], -1)
        return struct_reconstructed, x_hat

    def forward(self, g, h):
        # encode
        for layer in self.shared_encoder:
            h = layer(g, h).view(h.shape[0], -1)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(h)
        # decode feature matrix
        x_hat = self.attr_decoder(g, h).view(h.shape[0], -1)
        # return reconstructed matrices
        return struct_reconstructed, x_hat


class AnomalyDAE(nn.Module):
    def __init__(self, feat_size, num_node, hidden_size1, hidden_size2,
                 num_layer=2, num_head=4, dropout=0.5, activation=None) -> None:
        super(AnomalyDAE, self).__init__()

        self.num_node = num_node 
        self.hidden1 = nn.Sequential(
            nn.Linear(feat_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gat_encoder = nn.ModuleList([
            GATConv(hidden_size1, hidden_size2//num_head, num_head),
            *[GATConv(hidden_size2, hidden_size2//num_head, num_head) for _ in range(num_layer-1)]
        ])

        self.hidden2 = nn.Sequential(
            nn.Linear(num_node, hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Dropout(dropout)
        )

    def forward(self, g, h):
        h_x = self.hidden1(h)
        for layer in self.gat_encoder:
            h_x = layer(g, h_x).view(self.num_node, -1)
        struct_emb = h_x
        feat_emb = self.hidden2(h.transpose(0, 1))
        # A_hat = Z_a \cdot Z_a ^ T
        struct_hat = torch.sigmoid(torch.mm(struct_emb, struct_emb.transpose(0, 1)))
        # X_hat = Z_a \cdot Z_x ^ T
        feat_hat = torch.sigmoid(torch.mm(struct_emb, feat_emb.transpose(0, 1)))
        return struct_hat, feat_hat