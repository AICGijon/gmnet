"""HistNet implementation. It contains actual HistNet code."""


import torch
from dlquantification.quantmodule.other.GMLayer import GMLayer
from dlquantification.quantmodule.other.MeanLayer import MeanLayer
from dlquantification.utils.utils import BaseBagGenerator
import torch.nn.functional as F
import geotorch
import numpy as np
import scipy

from dlquantification.dlquantification import DLQuantification
from dlquantification.utils.ckareg import CKARegularization


class Power(torch.nn.Module):
    def __init__(self, exponent):
        super(Power, self).__init__()
        self.exponent = exponent

    def forward(self, x):
        return torch.pow(x, self.exponent)


class GMNet_Module(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size_fe,
        dropout_fe,
        bag_size,
        device,
        num_gaussians,
        n_gm_layers,
        gaussian_dimensions,
        cka_regularization,
    ):
        super(GMNet_Module, self).__init__()
        if len(num_gaussians) != n_gm_layers:
            raise ValueError("num_gaussians should be a tuple of the same size as n_gm_layers")
        if len(gaussian_dimensions) != n_gm_layers:
            raise ValueError("gaussian_dimensions should be a tuple of the same size as n_gm_layers")

        if isinstance(hidden_size_fe, int):
            self.hidden_size_fe = [hidden_size_fe]
        else:
            self.hidden_size_fe = hidden_size_fe
        self.n_gm_layers = n_gm_layers
        self.num_gaussians = num_gaussians
        self.gaussian_dimensions = gaussian_dimensions
        self.cka_regularization = cka_regularization
        if n_gm_layers == 1:
            self.output_size = num_gaussians[0]
        else:
            self.output_size = 0
            for i in range(n_gm_layers):
                self.output_size = self.output_size + self.num_gaussians[i]

        if cka_regularization != 0 or cka_regularization != 'view':
            self.quant_regularization = True
            self.cka = CKARegularization()
            self.activations = [None] * self.n_gm_layers
            
            def getActivation(i):
                def hook(model, input, output):
                    self.activations[i] = output.view(-1,self.gaussian_dimensions[i])
                return hook
        else:
            self.quant_regularization = False
            

        self.gm_modules = torch.nn.ModuleList()
        for i in range(n_gm_layers):
            module = torch.nn.Sequential()
            if gaussian_dimensions[i] is not None:
                previous_size = input_size
                if self.hidden_size_fe is not None:
                    for j, layer_size in enumerate(self.hidden_size_fe):
                        module.add_module("hidden_gm_%d_%d" % (i,j), torch.nn.Linear(previous_size, layer_size))
                        module.add_module("fe_leaky_relu_%d_%d" % (i,j), torch.nn.LeakyReLU())
                        module.add_module("fe_dropout_%d_%d" % (i,j), torch.nn.Dropout(dropout_fe))
                        previous_size = layer_size
                #Last layer to the latent space where the gaussians work
                module.add_module("linear_gm_%d" % i, torch.nn.Linear(previous_size, gaussian_dimensions[i]))
            module.add_module("sigmoid_gm_%d" % i, torch.nn.Sigmoid())

            input_dimensions = gaussian_dimensions[i] if gaussian_dimensions[i] is not None else input_size

            if cka_regularization != 0 or cka_regularization=='view':
                # Forward hook to get the activations after the layer linear_gm_%d
                index = list(dict(module.named_children()).keys()).index('linear_gm_%d' % i)
                module[index].register_forward_hook(getActivation(i))

            gm_layer = GMLayer(
                n_features=input_dimensions,
                num_gaussians=num_gaussians[i],
                device=device,
            )
            geotorch.positive_definite(gm_layer, "covariance")

            centers = gm_layer.centers.detach().numpy()
            distances = scipy.spatial.distance.cdist(centers, centers)
            np.fill_diagonal(distances, np.inf)
            cov = np.power(np.mean(np.min(distances, axis=1)) / 2, 2)

            gm_layer.covariance = torch.eye(input_dimensions).repeat(num_gaussians[i], 1, 1) * cov
            module.add_module("gm_layer_%d" % i, gm_layer)
            module.add_module("gm_flatten_%i" % i, torch.nn.Flatten(start_dim=0, end_dim=1))
            module.add_module("gm_batch_norm_%d" % i, torch.nn.BatchNorm1d(num_features=num_gaussians[i]))
            module.add_module("gm_unflatten_%d" % i, torch.nn.Unflatten(0, (-1, bag_size)))
            self.gm_modules.append(module)

    def forward(self, input):
        output = []
        for gm_module in self.gm_modules:
            output.append(gm_module(input))
        output = torch.cat(output,dim=-1)
        return torch.mean(output,dim=1)
    
    def compute_regularization(self):
        return self.cka_regularization=="view" or self.cka_regularization!=0
    def apply_regularization(self):
        return self.cka_regularization!=0 and self.cka_regularization!="view"
    
    def get_regularization_term(self):
        if self.cka_regularization!='view':
            return self.cka_regularization*self.cka.feature_space_linear_cka(self.activations)
        else:
            return self.cka.feature_space_linear_cka(self.activations)

    def get_regularization_multiplier(self):
        if self.cka_regularization!='view':
            return self.cka_regularization
        else:
            return 1

    def get_parameters_to_log(self):
        return {
            "n_gm_layers": self.n_gm_layers,
            "num_gaussians": self.num_gaussians,
            "cka_regularization": self.cka_regularization,
            "hidden_size_fe": self.hidden_size_fe,
            "gaussian_dimensions": self.gaussian_dimensions,
        }


class GMNet(DLQuantification):
    """
    Class for using the HistNet quantifier.

    GMNet builds creates artificial samples with fixed size and learns from them. Every example in each sample goes
    through the network and we build a histogram with all the examples in a sample. This is used in the quantification
    module where we use this vector to quantify the sample.

    :param train_epochs: How many times to repeat the process of going over training data. Each epoch will train over
                         n_bags samples.
    :type train_epochs: int
    :param test_epochs: How many times to repeat the process over the testing data (returned prevalences are averaged).
    :type test_epochs: int
    :param start_lr: Learning rate for the network (initial value).
    :type start_lr: float
    :param end_lr: Learning rate for the network. The value will be decreasing after a few epochs without improving
                   (check patiente parameter).
    :type end_lr: float
    :param n_classes: Number of classes
    :type n_classes: int
    :param optimizer_class: torch.optim class to make the optimization. Example torch.optim.Adam
    :type optimizer_class: class
    :param lr_factor: Learning rate decrease factor after patience epochs have passed without improvement.
    :type lr_factor: float
    :param batch_size: Update weights after this number of samples.
    :type batch_size: int
    :param patience: Number of epochs after which we will decrease the learning rate if there is no improvement.
    :type patience: int
    :param n_bags: How many artificial samples to build per epoch. If we get a single value this is used for training,
                   val and test. If a tuple with three values is provided it will used as (n_bags_train,n_bags_val,
                   n_bags_test)
    :type n_bags: int or (int,int,int)
    :param bag_size: Number of examples per sample (train,val,test).
    :type bag_size: int or (int,int,int)
    :param bag_generator: Class that will be in charge of generating the samples.
    :type bag_generator: class
    :param val_bag_generator: Class that will be in charge of generating the validation samples.
    :type val_bag_generator: class
    :param test_bag_generator: Class that will be in charge of generating the test samples.
    :type test_bag_generator: class
    :param n_gm_layers: Number of GM layers
    :type n_gm_layers: int
    :param num_gaussians: Number of gaussians per GM layer. If it is an int, every layer will have the same number of
                        gaussians, but it could be a list or tuple of ints (with different or the same values) whose
                        length corresponds to the number of GM layers.
    :type num_gaussians: int or list/tuple of ints
    :param random_seed: Seed to make results reproducible. This net needs to generate the bags so the seed is important.
    :type random_seed: int
    :param dropout: Dropout to use in the network (avoid overfitting).
    :type dropout: float
    :param weight_decay: L2 regularization for the model.
    :type weight_decay: float
    :param val_split: By default we validate using the train data. If a split is given, we partition the data for using
                      it as validation and early stopping. We can receive the split in different ways: 1) float:
                      percentage of data reserved for validation. 2) int: if 0, training set is used as validation.
                      If any other number, this number of examples will be used for validation. 3) tuple: if we get a
                      tuple, this will be the specific indexes used for validation
    :type val_split: int, float or tuple
    :param quant_loss: loss function to optimize in the quantification problem. Classification loss if use_labels=True
                       is fixed (CrossEntropyLoss used)
    :type quant_loss: function
    :param epsilon: If the error is less than this number, do not update the weights in this iteration.
    :type epsilon: float
    :param feature_extraction_module: Pytorch module with the feature extraction layers.
    :type feature_extraction_module: torch.Module
    :param linear_sizes: Tuple or list with the sizes of the linear layers used in the quantification module.
    :type linear_sizes: tuple
    :param use_labels: If true, use the class labels to help fit the feature extraction module of the network. A mix of
                       quant_loss + CrossEntropyLoss will be used as the loss in this case.
    :type use_labels: boolean
    :param use_labels_epochs: After this number of epochs, do not use the labels anymore. By default is use_labels is
                              true, labels are going to be used for all the epochs.
    :type use_labels_epochs: int
    :param output_function: Output function to use. Possible values 'softmax' or 'normalize'. Both will end up with a
                            probability distribution adding one
    :type output_function: str
    :param num_workers: Number of workers to use in the dataloaders. Note that if you choose to use more than one worker
                        you will need to use device=torch.device('cpu') in the bag generators, if not, an exception
                        will be raised.
    :type num_workers: int
    :param use_fp16: If true, trains using half precision mode.
    :type use_fp16: boolean
    :param device: Device to use for training/testing.
    :type device: torch.device
    :param callback_epoch: Function to call after each epoch. Useful to optimize with Optuna
    :type callback_epoch: function
    :param save_model_path: File to save the model when trained. We also load it if exists to skip training.
    :type save_model_path: file
    :param save_checkpoint_epochs: Save a checkpoint every n epochs. This parameter needs save_model_path to be set as
    it reuses the name of the file but appending the extension ckpt to it.
    :type save_checkpoint_epochs: int
    :param tensorboard_dir: Path to a directory where to store tensorboard logs. We can explore them using
                            tensorboard --logdir directory. By default no logs are saved.
    :type tensorboard_dir: str
    :param use_wandb: If true, we use wandb to log the training.
    :type use_wandb: bool
    :param wandb_experiment_name: Name of the experiment in wandb.
    :type wandb_experiment_name: str
    :param log_samples: If true the network will log all the generated samples with p and p_hat and the loss (for
                        training and validation)
    :type log_samples:  bool
    :param verbose: Verbose level.
    :type verbose: int
    :param dataset_name: Only for logging purposes.
    :type dataset_name: str
    """

    def __init__(
        self,
        train_epochs,
        test_epochs,
        n_classes,
        start_lr,
        end_lr,
        n_bags,
        bag_size,
        random_seed,
        linear_sizes,
        feature_extraction_module,
        n_gm_layers,
        num_gaussians,
        gaussian_dimensions,
        batch_size: int,
        bag_generator: BaseBagGenerator,
        hidden_size_fe=None,
        dropout_fe=0,
        gradient_accumulation: int = 1,
        val_bag_generator: BaseBagGenerator = None,
        test_bag_generator: BaseBagGenerator = None,
        optimizer_class=torch.optim.AdamW,
        cka_regularization=0,
        dropout: float = 0,
        weight_decay: float = 0,
        lr_factor=0.1,
        val_split=0,
        quant_loss=torch.nn.L1Loss(),
        quant_loss_val=None,
        epsilon=0,
        output_function="softmax",
        metadata_size=None,
        use_labels: bool = False,
        use_labels_epochs=None,
        residual_connection=False,
        batch_size_fe=None,
        device=torch.device("cpu"),
        use_multiple_devices=False,
        patience: int = 20,
        num_workers: int = 0,
        use_fp16: bool = False,
        callback_epoch=None,
        save_model_path=None,
        save_checkpoint_epochs=None,
        verbose=0,
        tensorboard_dir=None,
        use_wandb: bool = False,
        wandb_experiment_name: str = None,
        log_samples=False,
        dataset_name="",
    ):
        torch.manual_seed(random_seed)

        # Init the model

        quantmodule = GMNet_Module(
            input_size=feature_extraction_module.output_size,
            bag_size=bag_size,
            device=device,
            num_gaussians=num_gaussians,
            n_gm_layers=n_gm_layers,
            gaussian_dimensions=gaussian_dimensions,
            hidden_size_fe=hidden_size_fe,
            dropout_fe=dropout_fe,
            cka_regularization=cka_regularization,
        )

        super().__init__(
            train_epochs=train_epochs,
            test_epochs=test_epochs,
            n_classes=n_classes,
            start_lr=start_lr,
            end_lr=end_lr,
            n_bags=n_bags,
            bag_size=bag_size,
            random_seed=random_seed,
            batch_size=batch_size,
            quantmodule=quantmodule,
            bag_generator=bag_generator,
            val_bag_generator=val_bag_generator,
            test_bag_generator=test_bag_generator,
            optimizer_class=optimizer_class,
            weight_decay=weight_decay,
            lr_factor=lr_factor,
            val_split=val_split,
            quant_loss=quant_loss,
            quant_loss_val=quant_loss_val,
            batch_size_fe=batch_size_fe,
            gradient_accumulation=gradient_accumulation,
            feature_extraction_module=feature_extraction_module,
            linear_sizes=linear_sizes,
            dropout=dropout,
            epsilon=epsilon,
            output_function=output_function,
            metadata_size=metadata_size,
            use_labels=use_labels,
            use_labels_epochs=use_labels_epochs,
            residual_connection=residual_connection,
            device=device,
            use_multiple_devices=use_multiple_devices,
            patience=patience,
            num_workers=num_workers,
            use_fp16=use_fp16,
            callback_epoch=callback_epoch,
            save_model_path=save_model_path,
            save_checkpoint_epochs=save_checkpoint_epochs,
            verbose=verbose,
            tensorboard_dir=tensorboard_dir,
            use_wandb=use_wandb,
            wandb_experiment_name=wandb_experiment_name,
            log_samples=log_samples,
            dataset_name=dataset_name,
        )
