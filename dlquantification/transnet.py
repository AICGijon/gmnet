"""HistNet implementation. It contains actual HistNet code."""

import torch
from dlquantification.utils.utils import BaseBagGenerator
from dlquantification.dlquantification import DLQuantification
from dlquantification.quantmodule.transformers.models import SetTransformer


class TransNetTransformers(torch.nn.Module):
    def __init__(
        self,
        n_classes,
        trans_input_size,
        trans_output_size=512,
        trans_ind_points=16,
        trans_hidden_dim=128,
        trans_heads=4,
    ):
        super(TransNetTransformers, self).__init__()
        self.transformer_model = SetTransformer(
            dim_input=trans_input_size,
            dim_output=trans_output_size,
            num_inds=trans_ind_points,
            dim_hidden=trans_hidden_dim,
            num_outputs=n_classes,
            num_heads=trans_heads,
        )
        self.output_size = trans_output_size * n_classes

    def forward(self, input):
        out = self.transformer_model(input)
        out = out.view(out.shape[0], -1)
        return out


class TransNet(DLQuantification):
    """
    Class for using the HistNet quantifier.

    HistNet builds creates artificial samples with fixed size and learns from them. Every example in each sample goes
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
    :param n_bins: Number of bins used to build the histogram.
    :type n_bins: int
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
    :param histogram: Which histogram to use (sigmoid, soft, softrbf, hard)
    :type histogram: str
    :param quantiles: If true, use a quantile version of the histogram.
    :type quantiles: boolean
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
        batch_size: int,
        bag_generator: BaseBagGenerator,
        gradient_accumulation: int = 1,
        val_bag_generator: BaseBagGenerator = None,
        test_bag_generator: BaseBagGenerator = None,
        optimizer_class=torch.optim.AdamW,
        dropout: float = 0,
        weight_decay: float = 0,
        lr_factor=0.1,
        val_split=0,
        quant_loss=torch.nn.L1Loss(),
        epsilon=0,
        trans_out_size=512,
        trans_ind_points=16,
        trans_hidden_dim=128,
        trans_heads=4,
        output_function="softmax",
        batch_normalization=False,
        batch_size_fe=None,
        use_labels: bool = False,
        use_labels_epochs=None,
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
        quantmodule = TransNetTransformers(
            n_classes=n_classes,
            trans_input_size=feature_extraction_module.output_size,
            trans_output_size=trans_out_size,
            trans_ind_points=trans_ind_points,
            trans_hidden_dim=trans_hidden_dim,
            trans_heads=trans_heads,
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
            feature_extraction_module=feature_extraction_module,
            dropout=dropout,
            linear_sizes=linear_sizes,
            batch_size_fe=batch_size_fe,
            quantmodule=quantmodule,
            gradient_accumulation=gradient_accumulation,
            bag_generator=bag_generator,
            val_bag_generator=val_bag_generator,
            test_bag_generator=test_bag_generator,
            optimizer_class=optimizer_class,
            weight_decay=weight_decay,
            lr_factor=lr_factor,
            val_split=val_split,
            quant_loss=quant_loss,
            epsilon=epsilon,
            output_function=output_function,
            use_labels=use_labels,
            use_labels_epochs=use_labels_epochs,
            device=device,
            use_multiple_devices=use_multiple_devices,
            patience=patience,
            batch_normalization=batch_normalization,
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
