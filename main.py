#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import dgl
import scipy.sparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm
from datetime import datetime

from model import *
from data_util import *


def train_aegis(dataset='Amazon', cuda=True, epoch1=100, epoch2=100):
    t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
    sw = SummaryWriter('logs/aegis_%s_%s' % (dataset, t))
    adj, attr, label, _ = load_anomaly_detection_dataset(dataset)

    feat = torch.FloatTensor(attr)
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj))
    graph = dgl.add_self_loop(graph)

    in_dim = feat.size(1)
    hidden_dim, z_dim = 64, 100

    gdn_ae = GDN_Autoencoder(in_dim, hidden_dim, layer='gcn')
    generator = Generator(hidden_dim, hidden_dim=32, out_dim=hidden_dim)
    discriminator = Discriminator(in_dim=hidden_dim, hidden_dim=32)

    if cuda:
        device = torch.device('cuda')
        graph = graph.to(device)
        feat = feat.to(device)
        gdn_ae = gdn_ae.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    else:
        device = torch.device('cpu')

    ae_epoch, gan_epoch = epoch1, epoch2
    lr1, lr2 = 5e-3, 3e-3
    
    # first train GDN-AE
    train_graph_ae(gdn_ae, lr1, ae_epoch, graph, feat, sw)
    real_emb = gdn_ae.encoder(graph, feat)
    z_hat = gdn_ae(graph, feat)
    # then train Ano-GAN
    train_ano_gan(generator, discriminator, lr2, gan_epoch, real_emb, device, sw)
    with torch.no_grad():
        score = discriminator(real_emb).view(-1)
        score = score.cpu().numpy()
        print('AUC: %.3f' % roc_auc_score(label, score))
        print('AUC: %.3f'% roc_auc_score(label, 1-score))
        score = torch.square(z_hat-feat).sum(1).cpu().numpy()
        print('AUC: %.3f' % roc_auc_score(label, score))
    fpr, tpr, _ = roc_curve(label, score)
    np.save('aegis_fpr', fpr)
    np.save('aegis_tpr', tpr)


def train_graph_ae(model, lr, epochs, graph, attr_feat, writer):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    criterion = nn.MSELoss()

    print('**********Train Graph Autoencoder**********')
    for i in tqdm(range(epochs)):
        z_hat = model(graph, attr_feat)
        loss = criterion(attr_feat, z_hat)
        writer.add_scalar('train/loss', loss, i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_ano_gan(generator, discriminator, lr, epochs, real_emb, device, writer):
    criterion = nn.BCELoss()
    real_label, fake_label = 1, 0
    optG = torch.optim.Adam(
        generator.parameters(),
        lr=lr
    )
    optD = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr
    )

    print('**********Train Ano-GAN**********')
    for epoch in tqdm(range(epochs)):
        discriminator.zero_grad()
        batch_size, emb_size = real_emb.shape
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        ###### discriminator training ######
        # discriminator on real
        output = discriminator(real_emb).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward(retain_graph=True)
        D_on_real = output.mean().item()

        # discriminator on fake
        noise = torch.randn(batch_size, emb_size, device=device)
        fake = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_on_fake_1 = output.mean().item() 

        errD = errD_real + errD_fake
        optD.step()
        ###### end discriminator training ######

        ###### begin generator training ######
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_on_fake_2 = output.mean().item()
        optG.step()
        ##### end generator training ######

        writer.add_scalar('train/errD', errD, epoch)
        writer.add_scalar('train/errG', errG, epoch)
        writer.add_scalar('train/errDG1', D_on_fake_1, epoch)
        writer.add_scalar('train/errDG2', D_on_fake_2, epoch)


def train_classification(dataset, cuda=True, epochs=100, lr=1e-3):
    dataset = dgl.data.PubmedGraphDataset()
    graph = dataset[0]
    feat = graph.ndata['feat']
    label = graph.ndata['label']
    train_mask, test_mask = graph.ndata['train_mask'], graph.ndata['test_mask']
    label = (label == 0).long()
    in_dim, num_class = feat.size(1), len(label.unique())
    hidden_dim = 100
    model = NodeClassification(in_dim, hidden_dim, num_class, activation=F.leaky_relu)

    if cuda:
        device = torch.device('cuda')
        model = model.cuda()
        graph = graph.to(device)
        feat = feat.to(device)
        label = label.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
    sw = SummaryWriter('logs/classification_%s_%s' % (dataset.name, t))
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        logits = model(graph, feat)
        loss = criterion(logits[train_mask], label[train_mask])
        loss.backward()
        optimizer.step()
        # training nodes
        logits = logits.detach().cpu()
        label_pred, anomaly_score = logits.argmax(-1), logits[:, 1]
        acc = accuracy_score(label.cpu()[train_mask], label_pred[train_mask])
        f1 = f1_score(label.cpu()[train_mask], label_pred[train_mask])
        auc = roc_auc_score(label.cpu()[train_mask], anomaly_score[train_mask])
        sw.add_scalar('train/loss', loss, i)
        sw.add_scalar('train/acc', acc, i)
        sw.add_scalar('train/auc', auc, i)
        sw.add_scalar('train/f1', f1, i)
        # testing nodes
        acc = accuracy_score(label.cpu()[test_mask], label_pred[test_mask])
        f1 = f1_score(label.cpu()[test_mask], label_pred[test_mask])
        auc = roc_auc_score(label.cpu()[test_mask], anomaly_score[test_mask])
        sw.add_scalar('test/acc', acc, i)
        sw.add_scalar('test/f1', f1, i)
        sw.add_scalar('test/auc', auc, i)
    torch.save(model.state_dict(), 'saved/classification_%s.pth' % (t))


def loss_func(a, a_hat, x, x_hat, weight1=1, weight2=1, alpha=0.6, mask=1):
    # adjacency matrix reconstruction
    struct_weight = weight1
    struct_error_total = torch.sum(torch.square(torch.sub(a, a_hat)), axis=-1)
    struct_error_sqrt = torch.sqrt(struct_error_total) * mask
    struct_error_mean = torch.mean(struct_error_sqrt)
    # feature matrix reconstruction
    feat_weight = weight2
    feat_error_total = torch.sum(torch.square(torch.sub(x, x_hat)), axis=-1)
    feat_error_sqrt = torch.sqrt(feat_error_total) * mask
    feat_error_mean = torch.mean(feat_error_sqrt)
    loss =  (1 - alpha) * struct_error_sqrt + alpha * feat_error_sqrt
    # loss = struct_error_sqrt
    return loss, struct_error_mean, feat_error_mean


def train_dominant(dataset, cuda=True, epochs=100, lr=5e-3):
    adj, attrs, label, _ = load_anomaly_detection_dataset(dataset)
    rand_fpr, rand_npr, _ = roc_curve(label, np.zeros_like(label))
    np.save('rand_fpr', rand_fpr)
    np.save('rand_tpr', rand_npr)
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj)).add_self_loop()
    adj = torch.FloatTensor(adj)
    attrs = torch.FloatTensor(attrs)
    '''
    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0]
    adj = graph.adjacency_matrix().to_dense()
    attrs = graph.ndata['feat']
    '''
    feat_size, hidden_size = attrs.size(1), 64
    # feat_size, hidden_size = attrs.shape[1], 64
    model = Dominant(feat_size=feat_size, hidden_size=hidden_size)

    if cuda:
        device = torch.device('cuda')
        graph = graph.to(device)
        adj = adj.to(device)
        attrs = attrs.to(device)
        model = model.cuda()
    else:
        device = torch.device('cpu')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
    sw = SummaryWriter('logs/dominant_%s_%s' % (dataset, t))
    
    # traning
    model.train()
    print('########## Training ##########')
    eps = 1e-8
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        A_hat, X_hat = model(graph, attrs)
        loss, struct_loss, feat_loss = loss_func(adj, A_hat, attrs, X_hat, 1 , 1)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()
        sw.add_scalar('train/loss', l, epoch)
        sw.add_scalar('train/struct_loss', struct_loss, epoch)
        sw.add_scalar('train/feat_loss', feat_loss, epoch)
        score = loss.detach().cpu().numpy()
        # score = (score - score.min()) / (score.max() - score.min() + eps)
        sw.add_scalar('train/auc', roc_auc_score(label, score), epoch)
        for k in [50, 100, 200, 300]:
            sw.add_scalar('train/precision@%d' % k, precision_at_k(np.array(label), score, k), epoch)
    fpr, tpr, _ = roc_curve(label, score)
    print(roc_auc_score(label, score))
    np.save('dominant_fpr', fpr)
    np.save('dominant_tpr', tpr)


def train_anomalydae(dataset, cuda=True, epochs=300, lr=5e-3):
    # load original network data
    adj, attrs, label = load_network_dataset(dataset)
    # manually add anomalies
    adj, attrs, label = make_anomalies_v1(adj, attrs, label)
    num_node = adj.shape[0]
    graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj))
    adj = torch.FloatTensor(adj)
    attrs = torch.FloatTensor(attrs)

    feat_size, hidden_size1, hidden_size2= attrs.shape[1], 256, 128
    model = AnomalyDAE(feat_size=feat_size, num_node=num_node,
                       hidden_size1=hidden_size1, hidden_size2=hidden_size2)

    if cuda:
        device = torch.device('cuda')
        graph = graph.to(device)
        adj = adj.to(device)
        attrs = attrs.to(device)
        model = model.cuda()
    else:
        device = torch.device('cpu')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
    sw = SummaryWriter('runs/anomalydae_%s' % t)
    
    # traning
    model.train()
    print('########## Training ##########')
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        A_hat, X_hat = model(graph, attrs)
        loss, struct_loss, feat_loss = loss_func(adj, A_hat, attrs, X_hat)
        l = torch.mean(loss)
        l.backward()
        optimizer.step()
        sw.add_scalar('train/loss', l, epoch)
        sw.add_scalar('train/struct_loss', struct_loss, epoch)
        sw.add_scalar('train/feat_loss', feat_loss, epoch)
        score = loss.detach().cpu().numpy()
        score = (score - score.min()) / (score.max() - score.min())
        sw.add_scalar('train/roc_auc', roc_auc_score(label, score), epoch)

    # test
    model.eval()
    print('########## Testing ##########')
    for epoch in tqdm(range(10)):
        # load original network data
        adj, attrs, label = load_network_dataset(dataset)
        # manually add anomalies
        adj, attrs, label = make_anomalies_v1(adj, attrs, label)
        graph = dgl.from_scipy(scipy.sparse.coo_matrix(adj)).to(device)
        adj = torch.FloatTensor(adj).to(device)
        attrs = torch.FloatTensor(attrs).to(device)
        
        with torch.no_grad():
            A_hat, X_hat = model(graph, attrs)
            loss, struct_loss, feat_loss = loss_func(adj, A_hat, attrs, X_hat)
            l = loss.mean()
            sw.add_scalar('test/loss', l, epoch)
            sw.add_scalar('test/struct_loss', struct_loss, epoch)
            sw.add_scalar('test/feat_loss', feat_loss, epoch)
            score = loss.detach().cpu().numpy()
            score = (score - score.min()) / (score.max() - score.min())
            sw.add_scalar('test/roc_auc', roc_auc_score(label, score), epoch)


def train_triplet(dataset, cuda=True, epoch1=100, epoch2=50, lr=1e-3):
    # input attributed network G
    adj, attrs, label, _ = load_anomaly_detection_dataset(dataset)
    # create graph and attribute object, as anchor point
    graph_orig = dgl.from_scipy(scipy.sparse.coo_matrix(adj)).add_self_loop()
    attrs_orig = torch.FloatTensor(attrs)
    num_node, num_attr = attrs.shape
    # hidden dimension, output dimension
    hidden_dim, out_dim = 64, 32
    hidden_num = 2
    model = GRL(num_attr, hidden_dim, out_dim, hidden_num)
    criterion = nn.TripletMarginLoss()
    cuda_device = torch.device('cuda') if cuda else torch.device('cpu')
    cpu_device = torch.device('cpu')
    model = model.to(cuda_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
    sw = SummaryWriter('logs/triplet_%s_%s' % (dataset, t))
    model.train()
    # anomaly injection
    adj_aug, attrs_aug, label_aug = make_anomalies(adj, attrs, rate=.2, clique_size=15)
    # create triplet pairs
    pairs = set()
    for i in range(len(label_aug)):
        # if injected as anomaly, skip
        if label_aug[i] == 1: continue
        # i's neighbor that has been injected as an anomaly
        anomalous_nbs = np.nonzero(np.bitwise_and(adj[i].astype('bool'), label_aug.astype('bool')))[0]
        for nb in anomalous_nbs:
            pairs.add((i, nb))
    pairs = np.array(list(pairs))
    graph_aug = dgl.from_scipy(scipy.sparse.coo_matrix(adj_aug)).add_self_loop()
    attrs_aug = torch.FloatTensor(attrs_aug)
    label_aug = torch.LongTensor(label_aug)
    # copy to cuda device if applicable
    if cuda:
        graph_orig = graph_orig.to(cuda_device)
        attrs_orig = attrs_orig.to(cuda_device)
        graph_aug = graph_aug.to(cuda_device)
        attrs_aug = attrs_aug.to(cuda_device)
        label_aug = label_aug.to(cuda_device)

    # train encoder with supervised contrastive learning
    for i in tqdm(range(epoch1)):
        # augmented labels introduced by injection
        # train few shot learning
        orig = model.embed(graph_orig, attrs_orig)
        aug = model.embed(graph_aug, attrs_aug)
        margin_loss = criterion(orig[pairs[:, 0]], orig[pairs[:, 1]], aug[pairs[:, 1]])
        # bce_loss = criterion(logits, labels)
        sw.add_scalar('train/margin_loss', margin_loss, i)
        optimizer.zero_grad()
        margin_loss.backward()
        optimizer.step()
        # train reconstruction
        A_hat, X_hat = model(graph_aug, attrs_aug)
        a = graph_aug.adjacency_matrix().to_dense()
        recon_loss, struct_loss, feat_loss = loss_func(a.cuda() if cuda else a, A_hat, attrs_aug, X_hat, weight1=1, weight2=1, alpha=.7, mask=1-label_aug)
        recon_loss = recon_loss.mean()
        # loss = bce_loss + recon_loss
        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()
        sw.add_scalar('train/rec_loss', recon_loss, i)
        sw.add_scalar('train/struct_loss', struct_loss, i)
        sw.add_scalar('train/feat_loss', feat_loss, i)
    
    # evaluate
    with torch.no_grad():
        A_hat, X_hat = model(graph_orig, attrs_orig)
        a = graph_orig.adjacency_matrix().to_dense()
        recon_loss, struct_loss, feat_loss = loss_func(a.cuda() if cuda else a, A_hat, attrs_orig, X_hat, weight1=1, weight2=1, alpha=.7)
        score = recon_loss.detach().cpu().numpy()
        score = (score - score.min()) / (score.max() - score.min())
        print('AUC: %.4f' % roc_auc_score(label, score))
        for k in [50, 100, 200, 300]:
            print('Precision@%d: %.4f' % (k, precision_at_k(label, score, k)))
    fpr, tpr, _ = roc_curve(label, score)
    np.save('triplet_fpr', fpr)
    np.save('triplet_tpr', tpr)


def train_siamese(dataset, cuda=True, epoch1=100, epoch2=50, lr=1e-3, margin=0.5):
    # input attributed network G
    adj, attrs, label, _ = load_anomaly_detection_dataset(dataset)
    # create graph and attribute object, as anchor point
    graph1 = dgl.from_scipy(scipy.sparse.coo_matrix(adj)).add_self_loop()
    attrs1 = torch.FloatTensor(attrs)
    num_attr = attrs.shape[1]
    # hidden dimension, output dimension
    hidden_dim, out_dim = 128, 64
    hidden_num = 2
    model = GRL(num_attr, hidden_dim, out_dim, hidden_num)
    criterion = lambda z, z_hat, l: torch.square(z - z_hat) * (l==0) - l * torch.square(z - z_hat) + margin
    cuda_device = torch.device('cuda') if cuda else torch.device('cpu')
    cpu_device = torch.device('cpu')
    model = model.to(cuda_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
    sw = SummaryWriter('logs/siamese_%s_%s' % (dataset, t))
    model.train()
    # anomaly injection
    adj_aug, attrs_aug, label_aug = make_anomalies(adj, attrs, rate=.2, clique_size=20, sourround=50)
    graph2 = dgl.from_scipy(scipy.sparse.coo_matrix(adj_aug)).add_self_loop()
    attrs2 = torch.FloatTensor(attrs_aug)
    # train encoder with supervised contrastive learning
    for i in tqdm(range(epoch1)):
        # augmented labels introduced by injection
        labels = torch.FloatTensor(label_aug).unsqueeze(-1)
        if cuda:
            graph1 = graph1.to(cuda_device)
            attrs1 = attrs1.to(cuda_device)
            graph2 = graph2.to(cuda_device)
            attrs2 = attrs2.to(cuda_device)
            labels = labels.to(cuda_device)
        
        # train siamese loss
        orig = model.embed(graph1, attrs1)
        aug = model.embed(graph2, attrs2)
        margin_loss = criterion(orig, aug, labels)
        margin_loss = margin_loss.mean()
        sw.add_scalar('train/margin_loss', margin_loss, i)
        optimizer.zero_grad()
        margin_loss.backward()
        optimizer.step()
        
        # train reconstruction
        A_hat, X_hat = model(graph2, attrs2)
        a = graph1.adjacency_matrix().to_dense()
        recon_loss, struct_loss, feat_loss = loss_func(a.cuda() if cuda else a, A_hat, attrs1, X_hat, weight1=1, weight2=1, alpha=.7, mask=1)
        recon_loss = recon_loss.mean()
        # loss = bce_loss + recon_loss
        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()
        sw.add_scalar('train/rec_loss', recon_loss, i)
        sw.add_scalar('train/struct_loss', struct_loss, i)
        sw.add_scalar('train/feat_loss', feat_loss, i)
    
    # evaluate
    with torch.no_grad():
        A_hat, X_hat = model(graph1, attrs1)
        A_hat, X_hat = A_hat.cpu(), X_hat.cpu()
        a = graph1.adjacency_matrix().to_dense().cpu()
        recon_loss, struct_loss, feat_loss = loss_func(a, A_hat, attrs1.cpu(), X_hat, weight1=1, weight2=1, alpha=.7)
        score = recon_loss.detach().numpy()
        print('AUC: %.4f' % roc_auc_score(label, score))
        for k in [50, 100, 200, 300]:
            print('Precision@%d: %.4f' % (k, precision_at_k(label, score, k)))


if __name__ == '__main__':
    dataset = 'Flickr'
    # train_aegis(dataset=dataset, cuda=True, epoch1=200, epoch2=100)
    # train_dominant(dataset, epochs=200, lr=3e-3)
    train_siamese(dataset=dataset, cuda=True, epoch1=200)
    # train_triplet(dataset=dataset, cuda=True, epoch1=200)
    # train_anomalydae(dataset, epochs=300, lr=3e-4)