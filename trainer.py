import logging
import sys
import time

from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle, chain

import numpy as np
import torch

from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.optim import RMSprop, Adam, SGD
from tqdm import trange

from posterior import Posterior
from tensorboardX import SummaryWriter

def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False
def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True



logger = logging.getLogger(__name__)




class Trainer:
    default_metrics_to_monitor = []

    def __init__(
        self,
        args,
        model,
        gene_dataset,
        use_cuda=True,
        metrics_to_monitor=None,
        benchmark=False,
        frequency=None,
        weight_decay=1e-6,
        early_stopping_kwargs=None,
        data_loader_kwargs=None,
        show_progbar=True,
        seed=0,
        reconstruction_loss = 'alpha-gan',
        batch_size=128
    ):
        # handle mutable defaults
        early_stopping_kwargs = (
            early_stopping_kwargs if early_stopping_kwargs else dict()
        )
        data_loader_kwargs = data_loader_kwargs if data_loader_kwargs else dict()

        self.args = args
        self.model = model
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()
        self.seed = seed
        self.batch_size = batch_size

        self.data_loader_kwargs = {"batch_size": self.batch_size, "pin_memory": use_cuda}
        self.data_loader_kwargs.update(data_loader_kwargs)
        
        self.writer = SummaryWriter('result/logs')


        self.weight_decay = weight_decay
        self.benchmark = benchmark
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.training_time = 0

        if metrics_to_monitor is not None:
            self.metrics_to_monitor = set(metrics_to_monitor)
        else:
            self.metrics_to_monitor = set(self.default_metrics_to_monitor)

        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        self.frequency = frequency if not benchmark else None

        self.history = defaultdict(list)

        self.best_state_dict = self.model.state_dict()
        self.best_epoch = self.epoch

        self.show_progbar = show_progbar

    @torch.no_grad()
    def compute_metrics(self):
        begin = time.time()
        epoch = self.epoch + 1
        if self.frequency and (
            epoch == 0 or epoch == self.n_epochs or (epoch % self.frequency == 0)
        ):
            with torch.set_grad_enabled(False):
                self.model.eval()
                logger.debug("\nEPOCH [%d/%d]: " % (epoch, self.n_epochs))

                for name, posterior in self._posteriors.items():
                    message = " ".join([s.capitalize() for s in name.split("_")[-2:]])
                    if posterior.nb_cells < 5:
                        logging.debug(
                            message + " is too small to track metrics (<5 samples)"
                        )
                        continue
                    if hasattr(posterior, "to_monitor"):
                        for metric in posterior.to_monitor:
                            if metric not in self.metrics_to_monitor:
                                logger.debug(message)
                                result = getattr(posterior, metric)()
                                self.history[metric + "_" + name] += [result]
                    for metric in self.metrics_to_monitor:
                        result = getattr(posterior, metric)()
                        self.history[metric + "_" + name] += [result]
                self.model.train()
        self.compute_metrics_time += time.time() - begin

    def train(self, n_epochs=20, batch_size=128, lr=1e-4, eps=0.01, decay_lr=0.75, params=None, writer=None):
        begin = time.time()
        self.model.train()

        optimizer_eg = torch.optim.Adam(chain(self.model.z_encoder.parameters(), 
                                                               self.model.l_encoder.parameters(),
                                                              self.model.decoder.parameters()), 
                                                         lr=1e-4, eps=eps, weight_decay=self.weight_decay)
        if self.args.discriminator == 1:
            optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=1e-4, eps=eps, weight_decay=self.weight_decay)
        elif self.args.discriminator == 2:
            #optimizer_d = torch.optim.Adam(chain(d.parameters() for d in self.model.discriminators), lr=1e-4, eps=eps, weight_decay=self.weight_decay)
            print("two discriminator")
            optimizer_d = torch.optim.Adam(chain(self.model.discriminator.parameters(), self.model.rna_discriminator.parameters()), lr=1e-4, eps=eps, weight_decay=self.weight_decay)
        
        elif self.args.discriminator == 3:
            #optimizer_d = torch.optim.Adam(chain(d.parameters() for d in self.model.discriminators), lr=1e-4, eps=eps, weight_decay=self.weight_decay)
            print("three discriminator")
            optimizer_d = torch.optim.Adam(chain(self.model.discriminator.parameters(), self.model.rna_discriminator.parameters(),self.model.chip_discriminator.parameters()), lr=1e-4, eps=eps, weight_decay=self.weight_decay)

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        #self.compute_metrics()
        global_step = 0

        with trange(n_epochs, desc="training", file=sys.stdout, disable=not self.show_progbar) as pbar:
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)
                #i = 0
                for tensors_list in self.data_loaders_loop():
                    if tensors_list[0][0].shape[0] < 3:
                        continue
                    loss = self.loss(*tensors_list)
                    
                    reconst_loss, g_loss, d_loss, z_rec_loss, g_batch_loss, d_batch_loss = loss
                    #print('loss', reconst_loss, g_loss, d_loss, z_rec_loss)

                    if self.epoch > 0:
                        for _ in range(1):
                            # autoencoder loss, freeze D, update E and G
                            self.model.z_encoder.requires_grad = True
                            self.model.l_encoder.requires_grad = True
                            self.model.decoder.requires_grad = True
                            if self.args.discriminator == 2:
                                self.model.discriminator.requires_grad = False
                                self.model.rna_discriminator.requires_grad = False
                            elif self.args.discriminator == 3:
                                self.model.discriminator.requires_grad = False
                                self.model.rna_discriminator.requires_grad = False
                                self.model.chip_discriminator.requires_grad = False
                            optimizer_eg.zero_grad()
                            #print('g batch loss', reconst_loss, g_loss, z_rec_loss, g_batch_loss)
                            if g_batch_loss is not None:
                                (reconst_loss + g_loss + z_rec_loss + g_batch_loss*1000).backward(retain_graph=True)   # decoder to reconstruct
                                #(g_loss + g_batch_loss + z_rec_loss).backward(retain_graph=True) 
                            else:
                                #print('get total loss', reconst_loss, g_loss, z_rec_loss)
                                (reconst_loss+g_loss+z_rec_loss).backward(retain_graph=True)
                            optimizer_eg.step()

                    if self.epoch > 0:
                        for _ in range(1):
                            # discriminator loss, freeze E and G, update D
                            self.model.z_encoder.requires_grad = False
                            self.model.l_encoder.requires_grad = False
                            self.model.decoder.requires_grad = False
                            if self.args.discriminator == 2:
                                self.model.discriminator.requires_grad = True
                                self.model.rna_discriminator.requires_grad = True
                            elif self.args.discriminator == 3:
                                self.model.discriminator.requires_grad = True
                                self.model.rna_discriminator.requires_grad = True
                                self.model.chip_discriminator.requires_grad = True
                            optimizer_d.zero_grad()
                            if d_batch_loss is not None:
                                (d_loss + d_batch_loss*1000).backward()    # discriminate between fake and real
                            else:
                                d_loss.backward()
                            optimizer_d.step()
                    
                    #print(self.writer, self.epoch, global_step)
                    self.writer.add_scalar('reconst_loss', reconst_loss, global_step=global_step)
                    self.writer.add_scalar('g_loss', g_loss, global_step=global_step)
                    self.writer.add_scalar('d_loss', d_loss, global_step=global_step)
                    self.writer.add_scalar('z_rec_loss', z_rec_loss, global_step=global_step)
                    if d_batch_loss is not None:
                        self.writer.add_scalar('g_batch_loss', g_batch_loss, global_step=global_step)
                        self.writer.add_scalar('d_batch_loss', d_batch_loss, global_step=global_step)
                    global_step += 1


        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time
        if self.frequency:
            logger.debug(
                "\nTraining time:  %i s. / %i epochs"
                % (int(self.training_time), self.n_epochs)
            )

    def on_epoch_begin(self):
        pass

    @property
    @abstractmethod
    def posteriors_loop(self):
        pass

    def data_loaders_loop(self):  # returns an zipped iterable corresponding to loss signature
        data_loaders_loop = [self._posteriors[name] for name in self.posteriors_loop]
        #print('data loaders lop', data_loaders_loop[0], data_loaders_loop[1:])
        return zip(
            data_loaders_loop[0],
            *[cycle(data_loader) for data_loader in data_loaders_loop[1:]]
        )

    def register_posterior(self, name, value):
        name = name.strip("_")
        self._posteriors[name] = value

    def corrupt_posteriors(
        self, rate=0.1, corruption="uniform", update_corruption=True
    ):
        if not hasattr(self.gene_dataset, "corrupted") and update_corruption:
            self.gene_dataset.corrupt(rate=rate, corruption=corruption)
        for name, posterior in self._posteriors.items():
            self.register_posterior(name, posterior.corrupted())

    def uncorrupt_posteriors(self):
        for name_, posterior in self._posteriors.items():
            self.register_posterior(name_, posterior.uncorrupted())

    def __getattr__(self, name):
        if "_posteriors" in self.__dict__:
            _posteriors = self.__dict__["_posteriors"]
            if name.strip("_") in _posteriors:
                return _posteriors[name.strip("_")]
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        if name.strip("_") in self._posteriors:
            del self._posteriors[name.strip("_")]
        else:
            object.__delattr__(self, name)

    def __setattr__(self, name, value):
        if isinstance(value, Posterior):
            name = name.strip("_")
            self.register_posterior(name, value)
        else:
            object.__setattr__(self, name, value)

    def train_test_validation(
        self,
        model=None,
        gene_dataset=None,
        train_size=0.1,
        test_size=None,
        type_class=Posterior,
    ):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )
        n = len(gene_dataset.X)
        if train_size == 1.0:
            n_train = n
            n_test = 0
        else:
            n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
        
        random_state = np.random.RandomState(seed=self.seed)
        permutation = random_state.permutation(n)
        indices_test = permutation[:n_test]
        indices_train = permutation[n_test : (n_test + n_train)]
        indices_validation = permutation[(n_test + n_train) :]

        #print('indices_train', indices_train)
        return (
            self.create_posterior(
                model, gene_dataset, indices=indices_train, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, indices=indices_test, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, indices=indices_validation, type_class=type_class
            ),
        )

    def create_posterior(
        self,
        model=None,
        gene_dataset=None,
        shuffle=False,
        indices=None,
        type_class=Posterior,
    ):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )
        return type_class(
            model,
            gene_dataset,
            shuffle=shuffle,
            indices=indices,
            use_cuda=self.use_cuda,
            data_loader_kwargs=self.data_loader_kwargs,
        )
    
    
class UnsupervisedTrainer(Trainer):
    default_metrics_to_monitor = ["elbo"]

    def __init__(
        self,
        args,
        model,
        gene_dataset,
        train_size=0.8,
        test_size=None,
        n_epochs_kl_warmup=400,
        **kwargs
    ):
        super().__init__(args, model, gene_dataset, **kwargs)
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        if type(self) is UnsupervisedTrainer:
            self.train_set, self.test_set, self.validation_set = self.train_test_validation(
                model, gene_dataset, train_size, test_size
            )
            self.train_set.to_monitor = ["elbo"]
            self.test_set.to_monitor = ["elbo"]
            self.validation_set.to_monitor = ["elbo"]

    @property
    def posteriors_loop(self):
        return ["train_set"]

    def loss(self, tensors):  # tensors is local data
        #print('loss', self.n_epochs_kl_warmup, self.kl_weight)
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
        #print('model input', sample_batch, sample_batch.sum(axis=1))
        #return 
        loss = self.model(
            sample_batch, local_l_mean, local_l_var, batch_index
        )
        
        if len(loss) == 2:
            reconst_loss, kl_divergence = loss
            return torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        elif len(loss) > 2:
            reconst_loss, kl_divergence, g_loss, d_loss, z_rec_loss, g_batch_loss, d_batch_loss = loss
            return torch.mean(reconst_loss + self.kl_weight * kl_divergence), g_loss, d_loss, z_rec_loss, g_batch_loss, d_batch_loss
        

    def on_epoch_begin(self):
        if self.n_epochs_kl_warmup is not None:
            self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)
        else:
            self.kl_weight = 1.0
            #self.kl_weight = 0
        
        self.gan_weight = 1.0

