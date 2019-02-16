# -*- coding: utf-8 -*-
"""Functions for training and testing a network."""

import os
import json

from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml

from src.vrd_data_loader_class import VRDDataLoader
from src.utils.metric_utils import evaluate_relationship_recall
from src.utils.file_utils import load_annotations

with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


class VRDTrainTester():
    """Train and test utilities on VRD."""

    def __init__(self, net, net_name, use_cuda=CONFIG['use_cuda']):
        """Initiliaze train/test instance."""
        self.net = net
        self.net_name = net_name
        self.use_cuda = use_cuda

    def train(self, optimizer, criterion, scheduler=None,
              epochs=5, batch_size=32, val_batch_size=100,
              loss_sampling_period=50):
        """Train a neural network if it does not already exist."""
        # Check if the model is already trained
        print("Performing training for " + self.net_name)
        model_path_name = CONFIG['models_path'] + self.net_name + '.pt'
        if os.path.exists(model_path_name):
            self.net.load_state_dict(torch.load(model_path_name))
            print("Found existing trained model.")
            if self.use_cuda:
                self.net.cuda()
            else:
                self.net.cpu()
            return self.net
        # Settings and loading
        self.net.train()
        self.criterion = criterion
        data_loader = self._set_data_loaders(batch_size, val_batch_size)
        if self.use_cuda:
            self.net.cuda()
            if self.criterion is not None:
                self.criterion = self.criterion.cuda()
            data_loader.cuda()
            self.val_data_loader = self.val_data_loader.cuda()
        batches = data_loader.get_batches()

        # Main training procedure
        loss_history = []
        for epoch in range(epochs):
            if scheduler is not None:
                scheduler.step()
            accum_loss = 0
            for batch_cnt, batch_start in enumerate(batches):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize on batch data
                loss = self._compute_loss(data_loader, epoch, batch_start)
                loss.backward()
                optimizer.step()

                # Print loss statistics
                accum_loss += loss.item()
                if (batch_cnt + 1) % loss_sampling_period == 0:
                    accum_loss /= loss_sampling_period
                    val_loss = self._compute_validation_loss()
                    loss_history.append((batch_cnt + 1, accum_loss, val_loss))
                    print(
                        '[%d, %5d] loss: %.3f, validation loss: %.3f'
                        % (epoch, batch_cnt, accum_loss, val_loss)
                    )
                    accum_loss = 0
        torch.save(self.net.state_dict(), model_path_name)
        print('Finished Training')
        if any(loss_history):
            self.plot_loss(loss_history)
        return self.net

    def test(self, batch_size=100, test_mode='relationship'):
        """Test a neural network."""
        print("Testing %s on VRD." % (self.net_name))
        self.net.eval()
        data_loader = self._set_test_data_loader(
            batch_size=batch_size, test_mode=test_mode).eval()
        if self.use_cuda:
            self.net.cuda()
            data_loader.cuda()
        scores, boxes, labels = {}, {}, {}
        for batch in data_loader.get_batches():
            outputs = self._net_outputs(data_loader, 0, batch)
            filenames = data_loader.get_files(0, batch)
            scores.update({
                filename: np.array(score_vec)
                for filename, score_vec
                in zip(filenames, outputs.cpu().detach().numpy().tolist())
            })
            boxes.update(data_loader.get_boxes(0, batch))
            labels.update(data_loader.get_labels(0, batch))
        debug_scores = {
            filename: np.argmax(scores[filename])
            for filename in scores
        }
        annotations = load_annotations('test')
        debug_labels = {
            rel['filename'][:rel['filename'].rfind('.')]: rel['predicate_id']
            for anno in annotations
            for rel in anno['relationships']
        }
        debug_annos = {
            rel['filename'][:rel['filename'].rfind('.')]: (
                rel,
                scores[rel['filename'][:rel['filename'].rfind('.')]].tolist()
            )
            for anno in annotations
            for rel in anno['relationships']
            if rel['filename'][:rel['filename'].rfind('.')] in scores
        }
        with open(self.net_name + '.json', 'w') as fid:
            json.dump(debug_annos, fid)
        print(sum(
            1 for name in debug_scores
            if debug_scores[name] == debug_labels[name]))
        with open(self.net_name + '.txt', 'w') as fid:
            fid.write(json.dumps([
                name for name in debug_scores
                if debug_scores[name] == debug_labels[name]]))
        for mode in ['relationship', 'unseen', 'seen']:
            for keep in [1, 70]:
                print(
                    'Recall@50-100 (top-%d) %s:' % (keep, mode),
                    evaluate_relationship_recall(
                        scores, boxes, labels, keep, mode
                    )
                )

    def plot_loss(self, loss_history):
        """
        Plot training and validation loss.

        loss_history is a list of 4-element tuples, like
        (epoch, batch number, train_loss, val_loss)
        """
        train_loss = [loss for _, loss, _ in loss_history]
        validation_loss = [val_loss for _, _, val_loss in loss_history]
        min_batch = min(batch for batch, _, _ in loss_history)
        ticks = [
            '%d' % batch if batch == min_batch else ''
            for batch, _, _ in loss_history
        ]
        non_white_ticks = [t for t, tick in enumerate(ticks) if tick]
        for epoch, position in enumerate(non_white_ticks):
            ticks[position] = str(epoch + 1) if not (epoch % 3) else ''
        _, axs = plt.subplots()
        axs.plot(train_loss)
        axs.plot(validation_loss, 'orange')
        plt.xticks(range(len(train_loss)), ticks)
        plt.title(self.net_name + ' Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch - Batch Number')
        plt.legend(['Train Loss', 'Val. Loss'], loc='upper left')
        plt.savefig(
            CONFIG['figures_path'] + self.net_name + 'Loss.jpg',
            bbox_inches='tight'
        )

    def _compute_loss(self, data_loader, epoch, batch_start):
        """Compute loss for current batch."""
        loss = self.criterion(
            self._net_outputs(data_loader, epoch, batch_start),
            data_loader.get_targets(epoch, batch_start)
        )
        loss += sum(0.01 * param.norm(2) for param in self.net.parameters())
        return loss

    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.net.eval()
        batches = self.val_data_loader.get_batches()
        accum_loss = sum(
            self._compute_loss(self.val_data_loader, 0, batch_start).item()
            for batch_start in batches
        )
        self.net.train()
        return accum_loss / len(batches)

    def _set_data_loaders(self, batch_size, val_batch_size):
        """Set data loaders used during training."""
        data_loader = VRDDataLoader(batch_size=batch_size)
        self.val_data_loader = VRDDataLoader(batch_size=val_batch_size).eval()
        return data_loader

    def _set_test_data_loader(self, batch_size, test_mode):
        """Set data loader used during testing."""
        return VRDDataLoader(batch_size=batch_size, test_mode=test_mode)

    def _net_outputs(self, data_loader, epoch, batch_start):
        """Get network outputs for current batch."""
        return self.net(
            data_loader.get_union_boxes_pool5_features(epoch, batch_start)
        )
