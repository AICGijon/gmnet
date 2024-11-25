from dlquantification.utils.utils import BaseBagGenerator, UnlabeledMixerBagGenerator, APPBagGenerator
import torch


class LeQuaBagGenerator(BaseBagGenerator):
    def __init__(
        self,
        device,
        seed,
        prevalences,
        sample_size,
        app_bags_proportion,
        mixed_bags_proportion,
        labeled_unlabeled_split,
    ):
        self.device = device
        self.appBagGenerator = APPBagGenerator(device=device, seed=seed)
        self.unlabeledMixerBagGenerator = UnlabeledMixerBagGenerator(
            device=device,
            prevalences=prevalences,
            sample_size=sample_size,
            real_bags_proportion=1 - mixed_bags_proportion,
            seed=seed,
        )
        self.labeled_unlabeled_split = labeled_unlabeled_split
        self.mixed_bags_proportion = mixed_bags_proportion
        self.app_bags_proportion = app_bags_proportion
        self.seed = seed
        self.sample_size = sample_size
        self.labeled_indexes = labeled_unlabeled_split[0]
        self.unlabeled_indexes = labeled_unlabeled_split[1]

    def compute_bags(self, n_bags: int, bag_size: int, y):
        app_bags = round(n_bags * self.app_bags_proportion)
        bags_from_unlabeled = n_bags - app_bags

        samples_app_indexes, prevalences_app = self.appBagGenerator.compute_bags(
            n_bags=app_bags, bag_size=bag_size, y=y[self.labeled_indexes]
        )
        if bags_from_unlabeled > 0:
            samples_unlabeled_indexes, prevalences_unlabeled = self.unlabeledMixerBagGenerator.compute_bags(
                n_bags=bags_from_unlabeled, bag_size=bag_size
            )
            # The trick here is that the indexes of the second set of samples should be displaced to the right, because normal bag generators suppose
            # the first index is zero. In this case, the unlabeled data starts in 5000
            samples_unlabeled_indexes = torch.add(samples_unlabeled_indexes, len(self.labeled_indexes))
            samples_indexes = torch.cat((samples_app_indexes, samples_unlabeled_indexes))
            prevalences = torch.cat((prevalences_app, prevalences_unlabeled))
        else: 
            samples_indexes = samples_app_indexes
            prevalences = prevalences_app
        # Now mix all the samples an return them
        suffle = torch.randperm(n_bags)
        samples_indexes = samples_indexes[suffle, :]
        prevalences = prevalences[suffle, :]
        return samples_indexes, prevalences
    
    def get_parameters_to_log(self):
        return {'baggeneratorname':'LeQuaBagGenerator',
                'app_bags_proportion': self.app_bags_proportion,
                'mixed_bags_proportion': self.mixed_bags_proportion,
                'seed':self.seed,
                'sample_size':self.sample_size}
