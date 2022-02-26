import dgl
import os
import numpy as np
import networkx as nx
from scipy.sparse import data
import torch
import torch.nn.functional as F
import sklearn.preprocessing as preprocessing
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.linalg import inv, fractional_matrix_power


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g


def _rwr_trace_to_dgl_graph(
    g, seed, trace, positional_embedding_size, entire_graph=False
):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)

    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg


def batcher(batch):
    graph_q, graph_k = zip(*batch)
    graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
    return graph_q, graph_k


def labeled_batcher(batch):
    graph_q, label = zip(*batch)
    graph_q = dgl.batch(graph_q)
    return graph_q, torch.LongTensor(label)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))


def create_anomaly_detection_dataset(dataset):
    # load graphml as networkx graph
    nx_graph = nx.read_graphml(f'data/ad/{dataset}.graphml', node_type=int, force_multigraph=True)
    graph = dgl.DGLGraph()
    # convert to dgl graph
    graph.from_networkx(nx_graph)
    memo = {}
    with open(f'data/ad/{dataset}.true', 'r') as f:
        for line in f:
            # each line is: <node _ndex>;<label>
            idx, val = line.strip().split(';')
            memo[int(idx)] = int(val)
    labels = np.zeros(len(memo))
    for idx, val in memo.items():
        labels[idx] = val
    return graph, labels.astype('int32')


def compute_ppr(a, alpha=0.2, self_loop=True):
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def load_anomaly_detection_dataset(dataset, datadir='data'):
    datadir = os.path.join(datadir, dataset)

    if not os.path.exists(f'{datadir}/diff.npy'):
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        data_mat = scipy.io.loadmat(f'{datadir}/{dataset}.mat')
        try:
            adj = data_mat['A']
            feat = data_mat['X']
            truth = data_mat['gnd']
        except Exception:
            adj = data_mat['Network']
            feat = data_mat['Attributes']
            truth = data_mat['Label']
        truth = truth.flatten()
        # flatten sparse matrix
        if not isinstance(adj, np.ndarray):
            adj = adj.toarray()
        if not isinstance(feat, np.ndarray):
            feat = feat.toarray()
        diff = compute_ppr(adj, 0.2)
        
        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/truth.npy', truth)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        feat = np.load(f'{datadir}/feat.npy')
        truth = np.load(f'{datadir}/truth.npy')

    return adj, feat, truth, diff


def load_network_dataset(dataset):
    mat = scipy.io.loadmat(f'data/ad/{dataset}')
    try:
        adj = mat['Network'].toarray()
        feat = mat['Attributes'].toarray()
        label = mat['Label'].flatten() # this is the class label, not anomaly
    except:
        adj = mat['A']
        feat = mat['X']
        label = mat['gnd']
    '''
    try:
        diff = mat['diff'].toarray()
    except:
        diff = compute_ppr(adj, 0.2)
        mat['diff'] = scipy.sparse.coo_matrix(diff)
        scipy.io.savemat(f'data/ad/{dataset}', mat)
    '''
    return adj, feat, label


def graph_transform(adj, feat):
    adj_aug, feat_aug = adj.copy(), feat.copy()
    assert(adj_aug.shape[0]==feat_aug.shape[0])
    num_nodes = adj_aug.shape[0]
    for i in range(num_nodes):
        one_fifth = np.random.randint(0, 6)
        # 0.2 probability to perturb
        if one_fifth == 1:
            # do perturbation
            one_third = np.random.randint(0, 3)
            if one_third == 0:
                # add edge
                idxs = np.random.choice(np.arange(num_nodes), 3, replace=False)
                # add undirected edge
                for idx in idxs: 
                    adj_aug[i][idx] == 1
                    adj_aug[idx][i] == 1
            elif one_third == 1:
                # drop edge
                neighbors = np.nonzero(adj_aug[i])[0]
                idxs = np.random.choice(neighbors, size=min(len(neighbors), 3), replace=False)
                # add undirected edge
                for idx in idxs:
                    adj_aug[i][idx] = 0
                    adj_aug[idx][i] = 0
            elif one_third == 2:
                # swap attr
                neighbors = np.nonzero(adj_aug[i])[0]
                if neighbors.size:
                    idx = np.random.choice(neighbors, 1)
                    feat_aug[i], feat_aug[idx] = feat_aug[idx], feat_aug[i]
    return adj_aug, feat_aug


def make_anomalies(adj, feat, rate=.1, clique_size=30, sourround=50, scale_factor=10):
    adj_aug, feat_aug = adj.copy(), feat.copy()
    label_aug = np.zeros(adj.shape[0])
    assert(adj_aug.shape[0]==feat_aug.shape[0])
    num_nodes = adj_aug.shape[0]
    for i in range(num_nodes):
        prob = np.random.uniform()
        if prob > rate: continue
        label_aug[i] = 1
        one_fourth = np.random.randint(0, 4)
        if one_fourth == 0 or one_fourth == 1:
            # add clique
            new_neighbors = np.random.choice(np.arange(num_nodes), clique_size, replace=False)
            for n in new_neighbors:
                adj[n][i] = 1
                adj[i][n] = 1
        elif one_fourth == 2 or one_fourth == 3:
            # drop all connection
            neighbors = np.nonzero(adj[i])[0]
            for n in neighbors:
                adj[i][n] = 0
                adj[n][i] = 0
        elif one_fourth == 2:
            # attrs
            candidates = np.random.choice(np.arange(num_nodes), sourround, replace=False)
            max_dev, max_idx = 0, i
            for c in candidates:
                dev = np.square(feat[i]-feat[c]).sum()
                if dev > max_dev:
                    max_dev = dev
                    max_idx = c
            feat[i] = feat[max_idx]
        else:
            # scale attr
            prob = np.random.uniform(0, 1)
            if prob > 0.5:
                feat[i] *= scale_factor
            else:
                feat[i] /= scale_factor
    return adj_aug, feat_aug, label_aug


def make_anomalies_v1(adj, feat, label=None, m=15, k=10, n=50):
    '''
    add anomalies to original dataset:
    1. choose m nodes, make them fully connected to each other,
    repeat for k times;
    2. choose m nodes, for each one of them, sample n other nodes,
    change feature to the furthest of those n nodes', repeat
    for k times
    '''
    # cliques
    num_v = adj.shape[0]
    label_new = np.zeros(num_v)
    adj_new, feat_new = adj.copy(), feat.copy()
    for i in range(k):
        indices = np.random.choice(np.arange(num_v), size=m)
        for idx_i in indices:
            for idx_j in indices:
                adj_new[idx_i][idx_j] = adj_new[idx_j][idx_i] = 1
        label_new[indices] = 1
    
    # feat anomalies
    for i in range(k):
        indices = np.random.choice(np.arange(num_v), size=m)
        for idx in indices:
            candidates = np.random.choice(np.arange(num_v), size=n)
            cur_dist, cur_idx = 0, idx
            for c in candidates:
                tmp_dist = np.linalg.norm(feat[idx] - feat[c])
                if tmp_dist > cur_dist:
                    cur_dist = tmp_dist
                    cur_idx = c
            feat_new[idx] = feat[c]
        label_new[indices] = 1

    return adj_new, feat_new, label_new


def make_anomalies_v2(adj, feat, label, k1=20, k2=20, k3=10):
    '''
    add anomalies to original dataset:
    1. select k1 nodes in each class, change their feature to other class';
    2. select k2 nodes in each class, change their edges to other class';
    3. select k3 nodes in each class, make them fully connected
    '''
    k = k1 + k2 + k3
    label2idx = {l.item(): (label==l).nonzero()[0] for l in np.unique(label)}
    label2chosen = {}
    for l in label2idx.keys():
        chosen = np.random.choice((label==l).nonzero()[0], size=k)
        label2chosen[l] = chosen
    # step1: feature anomalies
    random_idx = np.random.permutation(np.arange(1, 7))
    feat_new = feat.copy()
    for i in label2chosen.keys():
        feat_new[label2chosen[i][:k1]] = feat[label2chosen[random_idx[i-1]][:k1]]
    # step2: community anomalies
    random_idx = np.random.permutation(np.arange(1, 7))
    adj_new = adj.copy()
    for i in label2chosen.keys():
        adj_new[label2chosen[i][k1:k2], :] = adj[label2chosen[random_idx[i-1]][k1:k2], :]
        adj_new[:, label2chosen[i][k1:k2]] = adj[:, label2chosen[random_idx[i-1]][k1:k2]]
    # step3: structural anomalies
    candidates = [val[k2:] for val in label2chosen.values()]
    candidates = np.stack(candidates).flatten()
    for u in candidates:
        for v in candidates:
            adj[u][v] = adj[v][u] = 1
    # step4: make the anomalies have label `1`
    label_new = np.zeros(feat.shape[0])
    for idx in label2chosen.values():
        label_new[idx] = 1
    
    return adj_new, feat_new, label_new

def make_anomaly_v3(adj, feat, prob=.1):
    raise NotImplementedError


def precision_at_k(truth, score, k):
    ranking = np.argsort(-score)    # higher scores ranked higher
    top_k = ranking[:k]
    top_k_label = truth[top_k]
    return top_k_label.sum() / k