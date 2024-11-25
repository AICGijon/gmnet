from torch.utils.data.sampler import Sampler
from quantificationlib.bag_generator import (
    PriorShift_BagGenerator,
    CovariateShift_BagGenerator,
    PriorAndCovariateShift_BagGenerator,
)
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import numpy as np


class BaseBagGenerator(ABC):
    """Base class for generating bags for histnet. Some bag generator are implemented for using labels,
    other can work without them."""

    @abstractmethod
    def compute_bags(self, n_bags, bag_size, y=None):
        pass


class TestBagGenerator(BaseBagGenerator):
    def __init__(self, device, n_examples, seed=2032):
        self.device = device
        self.n_examples = n_examples
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def compute_bags(self, n_bags, bag_size, y=None):
        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        for i in range(n_bags):
            samples_indexes[i, :] = torch.randint(
                0, self.n_examples, (bag_size,), generator=self.gen, device=self.device
            )
        return samples_indexes, None


class APPBagGenerator(BaseBagGenerator):
    """
    This is a bag generator following the APP protocol. It needs class information.
    The idea of this bag generator is to generate random prevalences of the classes.
    This will not be suitable for a problem like the plankton one (check SampleBasedBagGenerator),
    where some classes are more probable than others.
    """

    def __init__(self, device, seed=2032):
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.uses_labels = True

    def compute_bags(self, n_bags, bag_size, y):
        """Compute bags for training or prediction. If we do not have the class information, compute the bags without taking into account the prevalences"""
        with torch.no_grad():
            if not torch.is_tensor(y):
                y = torch.IntTensor(y).to(self.device)

            # Tensor to return the result. Each bag in a row.
            samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
            classes = torch.unique(y)
            n_classes = len(classes)

            prevalences = torch.zeros((n_bags, n_classes), device=self.device)
            for i in range(n_bags):
                low = round(bag_size * 0.01)
                high = round(bag_size * 0.99)

                ps = torch.randint(low, high, (n_classes - 1,), generator=self.gen, device=self.device)
                ps = torch.cat([ps, torch.tensor([0, bag_size], device=self.device)])
                ps = torch.sort(ps)[0]
                ps = ps[1:] - ps[:-1]  # Number of samples per class
                prevalences[i, :] = ps / bag_size
                already_generated = 0
                for n, p in zip(classes, ps):
                    if p != 0:
                        examples_class = torch.where(y == n)[0]
                        samples_indexes[i, already_generated : already_generated + p] = examples_class[
                            torch.randint(len(examples_class), (p,), generator=self.gen, device=self.device)
                        ]
                        already_generated += p
                suffle = torch.randperm(bag_size)
                samples_indexes[i, :] = samples_indexes[i, suffle]
            return samples_indexes, prevalences


class ProgressiveAPPBagGenerator(BaseBagGenerator):
    """
    This is a bag generator following the APP protocol. It needs class information.
    The idea of this bag generator is to generate random prevalences of the classes.
    This will not be suitable for a problem like the plankton one (check SampleBasedBagGenerator),
    where some classes are more probable than others.
    """

    def __init__(self, device, seed=2032):
        self.device = device
        self.gen = torch.Generator(device=device)
        self.n_classes = 2
        self.gen.manual_seed(seed)
        self.uses_labels = True

    def compute_bags(self, n_bags, bag_size, y):
        with torch.no_grad():
            if not torch.is_tensor(y):
                y = torch.IntTensor(y).to(self.device)

            # Tensor to return the result. Each bag in a row.
            samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
            classes = torch.unique(y)
            real_classes = len(classes)
            classes = classes[0 : self.n_classes]

            prevalences = torch.zeros((n_bags, real_classes), device=self.device)
            for i in range(n_bags):
                low = round(bag_size * 0.01)
                high = round(bag_size * 0.99)

                ps = torch.randint(low, high, (self.n_classes - 1,), generator=self.gen, device=self.device)
                ps = torch.cat([ps, torch.tensor([0, bag_size], device=self.device)])
                ps = torch.sort(ps)[0]
                ps = ps[1:] - ps[:-1]  # Number of samples per class
                prevalences[i, 0 : self.n_classes] = ps / bag_size
                already_generated = 0
                for n, p in zip(classes, ps):
                    if p != 0:
                        examples_class = torch.where(y == n)[0]
                        samples_indexes[i, already_generated : already_generated + p] = examples_class[
                            torch.randint(len(examples_class), (p,), generator=self.gen, device=self.device)
                        ]
                        already_generated += p
                suffle = torch.randperm(bag_size)
                samples_indexes[i, :] = samples_indexes[i, suffle]
            return samples_indexes, prevalences


class WindowBasedBagGenerator(BaseBagGenerator):
    """
    This bag generator considers a size of sample k and a stride s. It takes the labels of all the training
    samples and generates new artificial samples taking the first k examples. Then it displaces to the right, s
    examples a takes another k examples generating a new sample.
    Prior to take the examples, the samples touched by the window should be shuffled.
     This is repeated until the examples are finished.
    Considering that the number of examples is n, the number of bags is equal to n_bags = (n - (k - 1)) / s.
    """

    def __init__(
        self, device, samples_idxs, n_classes, identical_samples_each_time=False, stride=1, seed=2032, verbose=0
    ):
        self.device = device
        self.samples_idxs = samples_idxs
        self.stride = stride
        self.gen = torch.Generator(device=device)
        self.n_classes = n_classes
        self.gen.manual_seed(seed)
        self.seed = seed
        # create numpy random generator
        self.rng = np.random.RandomState(seed)
        self.identical_samples_each_time = identical_samples_each_time
        self.verbose = 0
        # we compute the bag of each sample
        self.ex_sample = np.hstack(np.asarray([len(s) * (i,) for i, s in enumerate(samples_idxs)]))

    def compute_bags(self, n_bags, bag_size, y=None):
        # compute the real number of bags, based in the parameters
        if self.identical_samples_each_time:
            # Fix the seed to generate always the same samples
            self.gen = torch.Generator(device=self.device)
            self.gen.manual_seed(self.seed)
            self.rng = np.random.RandomState(self.seed)

        n_examples = len(y)
        n_bags_computed = (n_examples - (bag_size - 1)) // self.stride
        assert n_bags_computed == n_bags
        if self.verbose > 0:
            print("Computing %d bags with a bag size of %d and a stride of %d" % (n_bags, bag_size, self.stride))
        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)
        sample_num = 0
        head = 0
        while head < (n_examples - bag_size - self.stride):
            # shuffle the samples that fall in this subsample
            samples_to_pick_from, counts = np.unique(self.ex_sample[head : (head + bag_size)], return_counts=True)
            n = 0
            for s, c in zip(samples_to_pick_from, counts):
                samples_indexes[sample_num, n : n + c] = torch.from_numpy(
                    self.rng.choice(self.samples_idxs[s], c, replace=False)
                )
                n += c

            prevalences[sample_num, :] = (
                torch.bincount(y[samples_indexes[sample_num, :]], minlength=self.n_classes) / bag_size
            )

            head += self.stride
            sample_num += 1

        return samples_indexes, prevalences


class SampleBasedBagGenerator(BaseBagGenerator):
    """
    This is a bag generator based on samples.
    We assume that we have a list of samples (with the prevalences) so we would like to
    generate similar ones for training an validation. Note that this bag generator
    **requires to have the class of each example**.
    """

    def __init__(
        self,
        device,
        prevalences_list,
        n_classes_to_change=0,
        same_sample=False,
        sample_idxs=None,
        all_samples=False,
        identical_samples_each_time=False,
        seed=2032,
        verbose=0,
    ):
        """Constructor

        Args:
            device (torch.device): device to use
            prevalences_list (np.array): Matrix with dimension (n_samples,n_classes) with the prevalence
            list of each sample
            n_classes_to_change (int, optional): Make prevalence changes in a random number of classes
            artificial sample. Defaults to 0.1.
            same_sample (bool, optional): After choosing a sample to get prevalences and make the variation, get
            the examples from this same sample. If false, examples can be drawn from the entire dataset
            all_samples (bool, optional): We will return all validation samples and not a random selection of samples
            sample_idxs (list, optional): If same_sample is true we need the sample indexes
            seed (int, optional): seed for random number generator. Defaults to 2032.
            verbose (int, optional): verbose value. Defaults to 0.
        """
        self.device = device
        self.seed = seed
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.rng = np.random.RandomState(seed)
        self.identical_samples_each_time = identical_samples_each_time
        self.prevalences_list = torch.Tensor(np.array(prevalences_list)).to(device)
        self.same_sample = same_sample
        self.sample_idxs = [torch.from_numpy(item) for item in sample_idxs] if sample_idxs is not None else None
        self.n_classes_to_change = n_classes_to_change
        self.all_samples = all_samples
        self.n_classes = self.prevalences_list.shape[1]
        self.n_samples = self.prevalences_list.shape[0]
        self.verbose = verbose

    def __compute_bags(self, n_bags, bag_size, y, prevalences_list):
        """[summary]
            Here we generate samples. There are two possible scenarios:
            1) We have class information. In this case we can do the following:
               1.1)Pick one sample randomly
               1.2)Pick one class randomly
               1.3)Generate a random change in the prevalence of that class
               1.4)Adjust the rest of the prevalences to sum 1
            2) We do not have class information. Think a solution.
        Args:
            n_bags (int): number of bags to generate
            bag_size (int): number of examples per bag
            y (np.array): vector with true labels for each example in the training set

        Returns:
            [np.array,np.array]: first array will be a matrix with dimensions (n_bags,bag_size) with the indexes
                                 of each sample second array will be a unidimensional array with the prevalence of each
                                 sample
        """
        with torch.no_grad():
            if not torch.is_tensor(y):
                y = torch.IntTensor(y).to(self.device)
            if y.device != self.device:
                y = y.to(self.device)

            # Tensor to return the result. Each bag in a row.
            samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
            prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)

            if self.identical_samples_each_time:
                # Fix the seed to generate always the same samples
                self.gen = torch.Generator(device=self.device)
                self.gen.manual_seed(self.seed)
                self.rng = np.random.RandomState(self.seed)

            # Generate the random stuff that we need for the method
            # Generate the samples that we are going to use
            if self.all_samples:
                if self.n_samples != n_bags:
                    raise ValueError("Number of bags do not correspond number of samples and all_samples is True")
                samples = torch.arange(self.n_samples, device=self.device)
            else:
                samples = torch.randint(len(prevalences_list), (n_bags,), generator=self.gen, device=self.device)

            if self.n_classes_to_change == 0 and self.same_sample:
                for i, sample in enumerate(samples):
                    samples_indexes[i, :] = torch.randint(0, len(self.sample_idxs[sample]), (bag_size,), generator=self.gen)
                    samples_indexes[i, :] = samples_indexes[i, :] + self.sample_idxs[sample][0]
                    prevalences[i, :] = torch.bincount(y[samples_indexes[i, :]], minlength=self.n_classes) / bag_size
                return samples_indexes, prevalences

            # Modify a random class to modify the prevalence for each sample
            classes_mod_sample = torch.randint(
                self.n_classes, (n_bags, self.n_classes_to_change), generator=self.gen, device=self.device
            )
            # Generate the amount of modification in the prevalence. The operation is for generating values in range
            # [-class_variation,class_variation]
            # variations_sample = (
            #    torch.rand((n_bags, self.n_classes_to_change), generator=self.gen, device=self.device)
            #    * self.class_variation
            #    * 2
            #    - self.class_variation
            # )
            variations_sample = (
                torch.rand((n_bags, self.n_classes_to_change), generator=self.gen, device=self.device) * 1.4 + 0.3
            )

            for i, (sample, classes_mod, variations) in enumerate(zip(samples, classes_mod_sample, variations_sample)):
                prevalences[i, :] = prevalences_list[sample]
                for class_mod, variation in zip(classes_mod, variations):
                    prevalences[i, class_mod] = torch.clip(prevalences[i, class_mod] ** variation, min=0, max=1)
                    # prevalences[i, class_mod] = torch.clip(prevalences[i, class_mod] * (1 + variation), min=0, max=1)

                # Rebalance the sample so the prevalences sum 1. The incremenet or decrement will be proportional to
                # the class prevalence.
                # mask = torch.ones(prevalences[i, :].numel(), dtype=torch.bool)
                # mask[classes_mod] = False
                # sum_other_classes_new_p = 1 - torch.sum(prevalences[i, classes_mod])
                # prevalences[i, mask] *= sum_other_classes_new_p / torch.sum(prevalences[i, mask])
                # make sure they add up to one
                prevalences[i, :] = F.normalize(prevalences[i, :], dim=0, p=1)

                # Prevalences are the probability of being picked for the sample. We need to divide the probability mass
                # for each by the number of examples of that class. That would be the prob of the example being picked
                if self.same_sample:
                    probs = (prevalences[i, :] / torch.bincount(y[self.sample_idxs[sample]], minlength=self.n_classes))[
                        y[self.sample_idxs[sample]]
                    ]
                    # In this case, we only get examples of the same sample. After getting the result of the multinomial
                    # we need to displace all the examples to the first element in the selected sample.
                    if bag_size == len(self.sample_idxs[sample]):
                        samples_indexes[i, :] = self.sample_idxs[sample]
                    else:
                        samples_indexes[i, :] = (
                            torch.from_numpy(
                                self.rng.choice(len(probs), bag_size, replace=True, p=probs.cpu().numpy())
                            ).to(self.device)
                            + self.sample_idxs[sample][0]
                        )

                else:
                    probs = (prevalences[i, :] / torch.bincount(y, minlength=self.n_classes))[y.long()]
                    # Carefull here with multinomial. It is not fully repdocible. Sometimes it gives a different index
                    # for one example, having the same number generator. Usually in the third call onwards.
                    # Do we need to remove this??
                    # samples_indexes[i, :] = torch.multinomial(probs, bag_size, replacement=True, generator=self.gen)
                    samples_indexes[i, :] = torch.from_numpy(
                        self.rng.choice(len(probs), bag_size, replace=True, p=probs.cpu().numpy())
                    ).to(self.device)

                # compute final prevalence
                prevalences[i, :] = torch.bincount(y[samples_indexes[i, :]], minlength=self.n_classes) / bag_size

            if self.verbose >= 1:
                print("Done.")
            return samples_indexes, prevalences

    def compute_bags(self, n_bags, bag_size, y):
        if self.verbose >= 1:
            print("Generating samples...", end="")
        return self.__compute_bags(n_bags, bag_size, y, self.prevalences_list)


class UnlabeledBagGenerator(BaseBagGenerator):
    """The idea of this bag generator is that we have samples with the prevalences but we do not have labels for each
    example. As we do not have labels, we cannot generate new samples but just use what we have.
    If pick_all is true we return all the available samples, if not we pick randomly (can be repeated)
    """

    def __init__(self, device, prevalences, sample_size, pick_all=False, seed=2032):
        self.device = device
        self.prevalences = prevalences
        self.sample_size = sample_size
        self.seed = seed
        self.pick_all = pick_all
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.n_classes = prevalences.shape[1]
        self.n_samples = prevalences.shape[0]
        self.uses_labels = False

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        """Compute training bags.

        Args:
            n_bags (int): number of bags to return (they are already generated)
            bag_size (int): size of the bag (this must match the size of the bags that we have). We assume bags having
            same size, but maybe this can change in the future.
            y ([type], optional): [description]. Defaults to None. No class information so this should be none

        Raises:
            ValueError: if there is an error in the parameteres
        """
        if bag_size != self.sample_size:
            raise ValueError("Given bag size must match bag sizes in the dataset")

        if self.pick_all and n_bags != self.n_samples:
            raise ValueError(
                "If you want to pick all the samples, the number of bags should match the number of bags available"
            )

        # We generate some random numbers to pick samples randomly
        if self.pick_all:
            sample_idxs = torch.arange(0, self.n_samples, device=self.device)
        else:
            sample_idxs = torch.randint(
                low=0, high=self.n_samples, size=(n_bags,), generator=self.gen, device=self.device
            )

        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)

        for i, bag_n in enumerate(sample_idxs):
            samples_indexes[i, :] = torch.arange(bag_n * bag_size, (bag_n + 1) * bag_size)
            prevalences[i, :] = self.prevalences[bag_n, :]

        return samples_indexes, prevalences


class UnlabeledMixerBagGenerator(BaseBagGenerator):
    """The idea here is to generate new samples from unlabeled ones. For that we mix N unlabeled samples and we pick
    randomly from the new sample. That will result in a sample with prevalences that will be the mean prevalence of the
    ones combined
    """

    def __init__(self, device, prevalences, sample_size, real_bags_proportion=0.5, seed=2032):
        self.device = device
        self.prevalences = prevalences
        self.sample_size = sample_size
        self.real_bags_proportion = real_bags_proportion
        self.seed = seed
        self.n_classes = prevalences.shape[1]
        self.n_samples = prevalences.shape[0]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.uses_labels=False

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        """Compute training bags.

        Args:
            n_bags (int): number of bags to return (they are already generated)
            bag_size (int): size of the bag (this must match the size of the bags that we have).
            We assume bags having same size, but maybe this can change in the future.
            y ([type], optional): [description]. Defaults to None. No class information so this should be none

        Raises:
            ValueError: if there is an error in the parameteres
        """
        # We generate some random numbers to pick samples randomly
        sample_idxs = torch.randint(low=0, high=self.n_samples, size=(n_bags,), generator=self.gen, device=self.device)

        real_bags_number = round(n_bags * self.real_bags_proportion)
        mix_bags_number = n_bags - real_bags_number

        # Generate samples to mix with
        sample_idxs_mix = torch.randint(
            low=0, high=self.n_samples, size=(mix_bags_number,), generator=self.gen, device=self.device
        )

        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)

        # Add real bags
        for i, bag_n in enumerate(sample_idxs[0:real_bags_number]):
            if bag_size == self.sample_size:
                samples_indexes[i, :] = torch.arange(bag_n * self.sample_size, (bag_n + 1) * self.sample_size)
            else:  # subsample or oversample
                samples_indexes[i, :] = torch.randint(
                    low=bag_n * self.sample_size,
                    high=(bag_n + 1) * self.sample_size,
                    size=(bag_size,),
                    generator=self.gen,
                    device=self.device,
                )

            prevalences[i, :] = self.prevalences[bag_n, :]

        for i, (bag_n_1, bag_n_2) in enumerate(
            zip(sample_idxs[real_bags_number:n_bags], sample_idxs_mix), start=real_bags_number
        ):
            mix = torch.cat(
                (
                    # TODO: revisar si esto está bien. No es un poco raro que tengamos bag_size aquí en lugar de
                    # sample size?
                    torch.arange(bag_n_1 * bag_size, (bag_n_1 + 1) * bag_size),
                    torch.arange(bag_n_2 * bag_size, (bag_n_2 + 1) * bag_size),
                )
            )
            samples_indexes[i, :] = mix[
                torch.randint(low=0, high=bag_size * 2, size=(bag_size,), generator=self.gen, device=self.device)
            ]
            prevalences[i, :] = torch.mean(
                torch.stack((self.prevalences[bag_n_1, :], self.prevalences[bag_n_2, :])), axis=0
            )

        # Suffle real and generated bags
        suffle = torch.randperm(n_bags)
        samples_indexes = samples_indexes[suffle, :]
        prevalences = prevalences[suffle, :]

        return samples_indexes, prevalences


class UnlabeledMixerBagGeneratorV2(BaseBagGenerator):
    """The idea of this bag generator is that the mixture between two samples is random"""

    def __init__(self, device, prevalences, sample_size, real_bags_proportion=0.5, seed=2032):
        self.device = device
        self.prevalences = prevalences
        self.sample_size = sample_size
        self.real_bags_proportion = real_bags_proportion
        self.seed = seed
        self.n_classes = prevalences.shape[1]
        self.n_samples = prevalences.shape[0]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.uses_labels=False

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        """Compute training bags.

        Args:
            n_bags (int): number of bags to return (they are already generated)
            bag_size (int): size of the bag (this must match the size of the bags that we have).
            We assume bags having same size, but maybe this can change in the future.
            y ([type], optional): [description]. Defaults to None. No class information so this should be none

        Raises:
            ValueError: if there is an error in the parameteres
        """
        # We generate some random numbers to pick samples randomly
        sample_idxs = torch.randint(
            low=0, high=self.n_samples, size=(n_bags, 2), generator=self.gen, device=self.device
        )
        # Generate the proportions in which we are going to mix the samples
        mixture = torch.rand(size=(n_bags,), generator=self.gen, device=self.device)

        # Real bags are bags with a mixture of 0 or 1
        real_bags_number = round(n_bags * self.real_bags_proportion)
        mixture[torch.randint(low=0, high=n_bags, size=(real_bags_number,), generator=self.gen, device=self.device)] = 0

        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)

        for i, (bags_idx, mix) in enumerate(zip(sample_idxs, mixture)):
            bag_n_1 = bags_idx[0]
            bag_n_2 = bags_idx[1]

            # print("Mixing bags %d and %d with proportion %.2f" % (bag_n_1, bag_n_2, mix))

            n_ex_bag1 = round((mix * bag_size).item())
            n_ex_bag2 = bag_size - n_ex_bag1

            # indexes for each of the samples
            bag_1_idxs = torch.arange(bag_n_1 * self.sample_size, (bag_n_1 + 1) * self.sample_size)
            bag_2_idxs = torch.arange(bag_n_2 * self.sample_size, (bag_n_2 + 1) * self.sample_size)

            # pick random elements for the new sample
            if mix == 0:
                samples_indexes[i, 0:bag_size] = bag_2_idxs
            elif mix == 1:
                samples_indexes[i, 0:bag_size] = bag_1_idxs
            else:
                samples_indexes[i, 0:n_ex_bag1] = bag_1_idxs[
                    torch.randint(
                        low=0, high=self.sample_size, size=(n_ex_bag1,), generator=self.gen, device=self.device
                    )
                ]
                samples_indexes[i, n_ex_bag1:bag_size] = bag_2_idxs[
                    torch.randint(
                        low=0, high=self.sample_size, size=(n_ex_bag2,), generator=self.gen, device=self.device
                    )
                ]

            prevalences[i, :] = self.prevalences[bag_n_1, :] * mix + self.prevalences[bag_n_2, :] * (1 - mix)

        return samples_indexes, prevalences


class UnlabeledMixerBagGeneratorV3(BaseBagGenerator):
    """The idea os this bag generator is that samples are mixed with all the elements on them. Is like joining
    two samples. The good point about this generator is that the prevalence of the new sample is exact."""

    def __init__(self, device, prevalences, sample_size, seed=2032):
        self.device = device
        self.prevalences = prevalences
        self.sample_size = sample_size
        self.seed = seed
        self.n_classes = prevalences.shape[1]
        self.n_samples = prevalences.shape[0]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.uses_labels=False

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        """Compute training bags.

        Args:
            n_bags (int): number of bags to return (they are already generated)
            bag_size (int): size of the bag (this must match the size of the bags that we have).
            We assume bags having same size, but maybe this can change in the future.
            y ([type], optional): [description]. Defaults to None. No class information so this should be none

        Raises:
            ValueError: if there is an error in the parameteres
        """
        if bag_size != self.sample_size * 2:
            raise ValueError(
                "ERROR: Bag size %d should be the double of sample size %d." % (bag_size, self.sample_size)
            )

        # We generate some random numbers to pick samples randomly
        sample_idxs = torch.randint(
            low=0, high=self.n_samples, size=(n_bags, 2), generator=self.gen, device=self.device
        )

        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)

        for i, bags_idx in enumerate(sample_idxs):
            bag_n_1 = bags_idx[0]
            bag_n_2 = bags_idx[1]

            # indexes for each of the samples
            bag_1_idxs = torch.arange(bag_n_1 * self.sample_size, (bag_n_1 + 1) * self.sample_size)
            bag_2_idxs = torch.arange(bag_n_2 * self.sample_size, (bag_n_2 + 1) * self.sample_size)

            # pick random elements for the new sample
            samples_indexes[i, 0 : self.sample_size] = bag_1_idxs
            samples_indexes[i, self.sample_size : self.sample_size * 2] = bag_2_idxs
            prevalences[i, :] = torch.mean(
                torch.stack((self.prevalences[bag_n_1, :], self.prevalences[bag_n_2, :])), axis=0
            )

        return samples_indexes, prevalences


class UnlabeledMixerBagGeneratorV4(BaseBagGenerator):
    """The idea here is to generate new samples from unlabeled ones. For that we mix N unlabeled samples and we pick
    randomly from the new sample. That will result in a sample with prevalences that will be the mean prevalence of the
    ones combined. The special idea of this bag generator is that samples that are nearer to each other get more
    probability of being combined. This will avoid centering the resulting bag prevalences in the center of the simplex
    """

    def __init__(self, device, prevalences, sample_size, real_bags_proportion=0.5, seed=2032):
        self.device = device
        self.prevalences = prevalences
        self.sample_size = sample_size
        self.real_bags_proportion = real_bags_proportion
        self.seed = seed
        self.n_classes = prevalences.shape[1]
        self.n_samples = prevalences.shape[0]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.uses_labels=False

        # Compute the distances from each prevalence to the rest
        distances = torch.cdist(self.prevalences, self.prevalences, p=1)
        self.probabilities = torch.zeros(distances.shape)
        for i in range(distances.shape[0]):
            d = torch.cat([distances[i, 0:i], distances[i, i + 1 :]])
            d = 1.0 / (d + 0.001)
            p = d / d.sum()
            self.probabilities[i, 0:i] = p[0:i]
            self.probabilities[i, i + 1 :] = p[i:]

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        """Compute training bags.

        Args:
            n_bags (int): number of bags to return (they are already generated)
            bag_size (int): size of the bag (this must match the size of the bags that we have).
            We assume bags having same size, but maybe this can change in the future.
            y ([type], optional): [description]. Defaults to None. No class information so this should be none

        Raises:
            ValueError: if there is an error in the parameteres
        """
        # We generate some random numbers to pick samples randomly
        sample_idxs = torch.randint(low=0, high=self.n_samples, size=(n_bags,), generator=self.gen, device=self.device)

        real_bags_number = round(n_bags * self.real_bags_proportion)

        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)

        # Add real bags
        for i, bag_n in enumerate(sample_idxs[0:real_bags_number]):
            if bag_size == self.sample_size:
                samples_indexes[i, :] = torch.arange(bag_n * self.sample_size, (bag_n + 1) * self.sample_size)
            else:  # subsample or oversample
                samples_indexes[i, :] = torch.randint(
                    low=bag_n * self.sample_size,
                    high=(bag_n + 1) * self.sample_size,
                    size=(bag_size,),
                    generator=self.gen,
                    device=self.device,
                )

            prevalences[i, :] = self.prevalences[bag_n, :]

        for i, bag_n_1 in enumerate(sample_idxs[real_bags_number:n_bags], start=real_bags_number):
            bag_n_2 = torch.multinomial(self.probabilities[bag_n_1, :], 1, generator=self.gen).item()
            mix = torch.cat(
                (
                    torch.arange(bag_n_1 * bag_size, (bag_n_1 + 1) * bag_size),
                    torch.arange(bag_n_2 * bag_size, (bag_n_2 + 1) * bag_size),
                )
            )
            samples_indexes[i, :] = mix[
                torch.randint(low=0, high=bag_size * 2, size=(bag_size,), generator=self.gen, device=self.device)
            ]
            prevalences[i, :] = torch.mean(
                torch.stack((self.prevalences[bag_n_1, :], self.prevalences[bag_n_2, :])), axis=0
            )

        # Suffle real and generated bags
        suffle = torch.randperm(n_bags)
        samples_indexes = samples_indexes[suffle, :]
        prevalences = prevalences[suffle, :]

        return samples_indexes, prevalences

class UnlabeledMixerBagGeneratorV5(BaseBagGenerator):
    def __init__(self, device, prevalences, sample_size, real_bags_proportion=0.5, seed=2032):
        self.device = device
        self.prevalences = prevalences
        self.sample_size = sample_size
        self.real_bags_proportion = real_bags_proportion
        self.seed = seed
        self.n_classes = prevalences.shape[1]
        self.n_samples = prevalences.shape[0]
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

        # Compute the distances from each prevalence to the rest
        distances = torch.cdist(self.prevalences, self.prevalences, p=1)
        self.probabilities = torch.zeros(distances.shape)
        for i in range(distances.shape[0]):
            d = torch.cat([distances[i, 0:i], distances[i, i + 1 :]])
            d = 1.0 / (d + 0.001)
            p = d / d.sum()
            self.probabilities[i, 0:i] = p[0:i]
            self.probabilities[i, i + 1 :] = p[i:]

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        # We generate some random numbers to pick samples randomly
        sample_idxs = torch.randint(low=0, high=self.n_samples, size=(n_bags,), generator=self.gen, device=self.device)

        real_bags_number = round(n_bags * self.real_bags_proportion)

        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)
        prevalences = torch.zeros((n_bags, self.n_classes), device=self.device)

        # TODO: Determine the adecuate value to consider "extreme", now <= 1/2*n_classes
        extreme_bags_index = (self.prevalences <= 1/(2*self.n_classes)).nonzero(as_tuple=True)[0]
        half_real_bags_number = round(real_bags_number/2)
        sample_idxs[half_real_bags_number: real_bags_number] = extreme_bags_index[torch.randint(low=0, high=extreme_bags_index.shape[0], size=(half_real_bags_number,))]

        # Add real bags
        for i, bag_n in enumerate(sample_idxs[0:real_bags_number]):
            if bag_size == self.sample_size:
                samples_indexes[i, :] = torch.arange(bag_n * self.sample_size, (bag_n + 1) * self.sample_size)
            else:  # subsample or oversample
                samples_indexes[i, :] = torch.randint(
                    low=bag_n * self.sample_size,
                    high=(bag_n + 1) * self.sample_size,
                    size=(bag_size,),
                    generator=self.gen,
                    device=self.device,
                )

            prevalences[i, :] = self.prevalences[bag_n, :]
        

        for i, bag_n_1 in enumerate(sample_idxs[real_bags_number:n_bags], start=real_bags_number):
            bag_n_2 = torch.multinomial(self.probabilities[bag_n_1, :], 1, generator=self.gen).item()
            mix = torch.cat(
                (
                    torch.arange(bag_n_1 * bag_size, (bag_n_1 + 1) * bag_size),
                    torch.arange(bag_n_2 * bag_size, (bag_n_2 + 1) * bag_size),
                )
            )
            samples_indexes[i, :] = mix[
                torch.randint(low=0, high=bag_size * 2, size=(bag_size,), generator=self.gen, device=self.device)
            ]
            prevalences[i, :] = torch.mean(
                torch.stack((self.prevalences[bag_n_1, :], self.prevalences[bag_n_2, :])), axis=0
            )

        # Suffle real and generated bags
        suffle = torch.randperm(n_bags)
        samples_indexes = samples_indexes[suffle, :]
        prevalences = prevalences[suffle, :]

        return samples_indexes, prevalences


class NoPrevalenceBagGenerator(BaseBagGenerator):
    """
    The idea of this bag generator is to use complete unlabeled samples, that is, no labels for individual samples
    but also no prevalences for the samples. We just know that they are real samples.
    This bag generator is meant to be used is some kind of semisupervised quantification learning.
    The bag returned will have the following form:
        || bag1_subsample_A | bag1_subsample_B | bag2_subsample ||
    The two subsamples should have approximatly the same prevalence, while the third subsample, a different prevalence
    """

    def __init__(
        self,
        device,
        sample_idxs=None,
        seed=2032,
        verbose=0,
    ):

        self.device = device
        self.seed = seed
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.rng = np.random.RandomState(seed)
        self.sample_idxs = [torch.from_numpy(item) for item in sample_idxs]
        self.n_samples = len(sample_idxs)
        self.verbose = verbose
        self.use_labels=False

    def generate_random_number_excluding(self, low, high, excluded):
        generated = False
        while not generated:
            value = torch.randint(low, high, (1,), generator=self.gen, device=self.device)
            if value != excluded:
                generated = True
        return value

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        if bag_size % 3 != 0:
            raise ValueError("Bag size should be divisible by three as we are returning three subbags on it")

        # Generate anchor samples
        anchor_idxs = torch.randint(low=0, high=self.n_samples, size=(n_bags,), generator=self.gen, device=self.device)

        samples_indexes = torch.zeros((n_bags, bag_size), dtype=torch.int64, device=self.device)

        for i, anchor in enumerate(anchor_idxs):
            positive = anchor
            negative = self.generate_random_number_excluding(low=0, high=self.n_samples, excluded=positive)

            individual_bag_size = bag_size // 3
            # pick random examples for anchor, positive and negative
            samples_indexes[i, 0:individual_bag_size] = self.sample_idxs[anchor][
                torch.randint(
                    low=0,
                    high=len(self.sample_idxs[anchor]),
                    size=(individual_bag_size,),
                    generator=self.gen,
                    device=self.device,
                )
            ]
            samples_indexes[i, individual_bag_size : 2 * individual_bag_size] = self.sample_idxs[positive][
                torch.randint(
                    low=0,
                    high=len(self.sample_idxs[positive]),
                    size=(individual_bag_size,),
                    generator=self.gen,
                    device=self.device,
                )
            ]
            samples_indexes[i, 2 * individual_bag_size : 3 * individual_bag_size] = self.sample_idxs[negative][
                torch.randint(
                    low=0,
                    high=len(self.sample_idxs[negative]),
                    size=(individual_bag_size,),
                    generator=self.gen,
                    device=self.device,
                )
            ]

        return samples_indexes, None


class QLibPriorShiftBagGenerator(BaseBagGenerator):
    def __init__(self, device, min_prevalence=None, seed=2032, method="Uniform", alphas=None):
        self.min_prevalence = min_prevalence
        self.seed = seed
        self.inner_generator = None
        self.method = method
        self.alphas = alphas
        self.device = device
        self.uses_labels=True

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        if self.inner_generator is None:
            self.inner_generator = PriorShift_BagGenerator(
                n_bags=n_bags,
                bag_size=bag_size,
                min_prevalence=self.min_prevalence,
                random_state=np.random.RandomState(self.seed),
                method=self.method,
                alphas=self.alphas,
            )

        prevalences, samples_indexes = self.inner_generator.generate_bags(X=None, y=y.cpu().numpy())
        prevalences = torch.from_numpy(np.transpose(prevalences)).to(self.device)
        return np.transpose(samples_indexes), prevalences


class QLibCovariateShiftBagGenerator(BaseBagGenerator):
    def __init__(self, device, X, seed=2032):
        self.seed = seed
        self.X = X
        self.inner_generator = None
        self.device = device

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        if self.inner_generator is None:
            self.inner_generator = CovariateShift_BagGenerator(n_bags=n_bags, bag_size=bag_size, random_state=self.seed)

        prevalences, samples_indexes = self.inner_generator.generate_bags(self.X, y.cpu().numpy())
        prevalences = torch.from_numpy(np.transpose(prevalences)).to(self.device)
        return np.transpose(samples_indexes), prevalences


class QLibBothShiftsBagGenerator(BaseBagGenerator):
    def __init__(self, device, X, min_prevalence=None, seed=2032):
        self.min_prevalence = min_prevalence
        self.seed = seed
        self.X = X
        self.inner_generator = None
        self.device = device

    def compute_bags(self, n_bags: int, bag_size: int, y=None):
        if self.inner_generator is None:
            self.inner_generator = PriorAndCovariateShift_BagGenerator(
                n_bags=n_bags,
                bag_size=bag_size,
                min_prevalence=self.min_prevalence,
                random_state=self.seed,
            )

        prevalences, samples_indexes = self.inner_generator.generate_bags(self.X, y.cpu().numpy())
        prevalences = torch.from_numpy(np.transpose(prevalences)).to(self.device)
        return np.transpose(samples_indexes), prevalences


class MRAEBagGenerator(BaseBagGenerator):
    def __init__(self, device, X, min_prevalence=None, seed=2032):
        self.device = device
        self.X = X
        self.seed = seed
        self.min_prevalence = min_prevalence
        self.inner_generator = None

    def compute_bags(self, n_bags: int, bag_size: int, y=None):

        normal_bags = round(n_bags * 0.5)
        prev_1_bags = round(n_bags * 0.25)
        prev_2_bags = n_bags - normal_bags - prev_1_bags

        if self.inner_generator is None:
            self.inner_generator_1 = PriorShift_BagGenerator(
                n_bags=normal_bags,
                bag_size=bag_size,
                random_state=self.seed,
            )

            self.inner_generator_2 = PriorShift_BagGenerator(
                n_bags=prev_1_bags,
                bag_size=bag_size,
                min_prevalence=[0, 1 - self.min_prevalence],
                random_state=self.seed,
            )

            self.inner_generator_3 = PriorShift_BagGenerator(
                n_bags=prev_2_bags,
                bag_size=bag_size,
                min_prevalence=[1 - self.min_prevalence, 0],
                random_state=self.seed,
            )

            samples_indexes = np.zeros((n_bags, bag_size), dtype=int)
            prevalences = np.zeros((n_bags, 2))

            prevalences_1, samples_indexes_1 = self.inner_generator_1.generate_bags(self.X, y.cpu().numpy())
            prevalences_1 = np.transpose(prevalences_1)
            samples_indexes_1 = np.transpose(samples_indexes_1)
            samples_indexes[0:normal_bags, :] = samples_indexes_1
            prevalences[0:normal_bags, :] = prevalences_1

            prevalences_2, samples_indexes_2 = self.inner_generator_2.generate_bags(self.X, y.cpu().numpy())
            prevalences_2 = np.transpose(prevalences_2)
            samples_indexes_2 = np.transpose(samples_indexes_2)
            samples_indexes[normal_bags : normal_bags + prev_1_bags, :] = samples_indexes_2
            prevalences[normal_bags : normal_bags + prev_1_bags, :] = prevalences_2

            prevalences_3, samples_indexes_3 = self.inner_generator_3.generate_bags(self.X, y.cpu().numpy())
            prevalences_3 = np.transpose(prevalences_3)
            samples_indexes_3 = np.transpose(samples_indexes_3)
            samples_indexes[normal_bags + prev_1_bags : normal_bags + prev_1_bags + prev_2_bags, :] = samples_indexes_3
            prevalences[normal_bags + prev_1_bags : normal_bags + prev_1_bags + prev_2_bags, :] = prevalences_3

            samples_indexes = torch.from_numpy(samples_indexes).to(self.device)
            prevalences = torch.from_numpy(prevalences).to(self.device)

            # Suffle real and generated bags
            suffle = torch.randperm(n_bags)
            samples_indexes = samples_indexes[suffle, :]
            prevalences = prevalences[suffle, :]

            return samples_indexes, prevalences


class BagSampler(Sampler):
    """This class is the integration with pytorch sampling.
    The idea is that this integrates well with the DataLoader."""

    def __init__(self, bagGenerator, n_bags=100, bag_size=100, batch_size=1, targets=None):
        """Constructor to initialize the sampler

        Args:
            bagGenerator (class): Bag generator to use
            targets (np.array): labels of the examples in the dataset. Can be none if this information does not exist.
            n_bags (int, optional): Number of bags to generate. Defaults to 100.
            bag_size (int, optional): Number of examples per bag. Defaults to 100.
        """
        self.bagGenerator = bagGenerator
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.targets = targets
        self.batch_size = batch_size

    def __iter__(self):
        """[summary]

        Returns:
            [iterator]: returns an iterator to iterate over the samples. Each sample is a set of indexes with the
            examples selected for each class
        """
        samples, self.p = self.bagGenerator.compute_bags(self.n_bags, self.bag_size, self.targets)
        position = 0
        while position < self.n_bags:
            if position + self.batch_size < len(samples):
                yield samples[position : position + self.batch_size].flatten()
            elif position < len(samples):
                # Last batch
                yield samples[position:].flatten()
            position += self.batch_size

    def __len__(self):
        if self.n_bags <= self.batch_size:
            return 1
        else:
            if self.n_bags % self.batch_size == 0:
                return self.n_bags // self.batch_size
            else:
                return (self.n_bags // self.batch_size) + 1


def batch_collate_fn(batch, bag_size, samples_ids=False):
    """This function receives a batch of samples, but without a batch structure (all the examples one after
    the other. It creates batches and returns it as a dictionary with x and y.

    TODO: we need to make this function generic. Now it is not.
    """
    batch_size = len(batch) // bag_size
    # for ifcb batch is list of tuples. Each tuple has three elements: the image, the label and the sample
    labels_present = isinstance(batch[0], tuple) and len(batch[0]) > 1
    x = [d[0] for d in batch]
    # Here it can happen that each element is a dict (for instance with bert)
    if type(x[0]) is dict:
        xdict = dict()
        for key in x[0].keys():
            xdict[key] = []
            for d in x:
                xdict[key].append(d[key])
            xdict[key] = torch.stack(xdict[key])
            xdict[key] = xdict[key].view(batch_size, bag_size, *xdict[key].shape[1:])
        x = xdict
    else:
        x = torch.stack(x)
        x = x.view(batch_size, bag_size, *x.shape[1:])
    if samples_ids:
        samples = [batch[i][2] for i in range(0, len(batch), bag_size)]
    if labels_present:
        y = [d[1] for d in batch]
        if type(y) is list:
            y = torch.tensor(y)
        else:
            y = torch.stack(y)
        y = y.view(batch_size, bag_size)
        # y = [y[n : n + bag_size] for n in range(0, len(y), bag_size)]
        if samples_ids:
            return {"x": x, "y": y, "samples_ids": samples}
        else:
            return {"x": x, "y": y}
    else:
        if samples_ids:
            return {"x": x, "samples_ids": samples}
        else:
            return {"x": x}
