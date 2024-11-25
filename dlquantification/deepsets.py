"""HistNet implementation. It contains actual HistNet code."""

import torch
from dlquantification.quantmodule.deepsets.AvgPooling import AvgPooling
from dlquantification.quantmodule.deepsets.MaxPooling import MaxPooling
from dlquantification.quantmodule.deepsets.MedianPooling import MedianPooling
from dlquantification.utils.utils import BaseBagGenerator

from dlquantification.dlquantification import DLQuantification


class DeepSetsQuantModule(torch.nn.Module):
    def __init__(self, input_size, pooling_layer):

        super(DeepSetsQuantModule, self).__init__()
        self.output_size = input_size
        # Create the quantification module for DeepSets
        if pooling_layer == "median":
            self.pooling_layer = MedianPooling()
        elif pooling_layer == "avg":
            self.pooling_layer = AvgPooling()
        elif pooling_layer == "max":
            self.pooling_layer = MaxPooling()
        else:
            raise ValueError("Invalid pooling layer type, expected one of [avg, median, max]")

        self.layers = torch.nn.Sequential()
        self.layers.add_module("pooling_layer", self.pooling_layer)

    def forward(self, input):
        return self.layers(input)


class DeepSets(DLQuantification):
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
        val_bag_generator: BaseBagGenerator = None,
        test_bag_generator: BaseBagGenerator = None,
        optimizer_class=torch.optim.AdamW,
        gradient_accumulation=1,
        dropout: float = 0,
        weight_decay: float = 0,
        lr_factor=0.1,
        val_split=0,
        quant_loss=torch.nn.L1Loss(),
        quant_loss_val=None,
        epsilon=0,
        pooling_layer="max",
        output_function="softmax",
        use_labels: bool = False,
        use_labels_epochs=None,
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
        batch_normalization=False,
        use_wandb: bool = False,
        wandb_experiment_name: str = None,
        log_samples=False,
        dataset_name="",
    ):
        torch.manual_seed(random_seed)

        # Init the model
        quantmodule = DeepSetsQuantModule(input_size=feature_extraction_module.output_size, pooling_layer=pooling_layer)

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
            feature_extraction_module=feature_extraction_module,
            linear_sizes=linear_sizes,
            dropout=dropout,
            epsilon=epsilon,
            output_function=output_function,
            gradient_accumulation=gradient_accumulation,
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
