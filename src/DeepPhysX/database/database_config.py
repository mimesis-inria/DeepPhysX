from typing import Optional
from os.path import isdir, sep, join, isabs, abspath


class DatabaseConfig:

    def __init__(self,
                 existing_dir: Optional[str] = None,
                 mode: Optional[str] = None,
                 shuffle: bool = True,
                 normalize: bool = True,
                 recompute_normalization: bool = False):
        """
        DatabaseConfig is a configuration class to parameterize the Database and the DatabaseManager.

        :param existing_dir: Path to an existing 'dataset' repository.
        :param mode: Specify the Dataset mode that should be used between 'training', 'validation' and 'running'.
        :param shuffle: Specify if the Dataset should be shuffled when a batch is taken.
        :param normalize: If True, the data will be normalized using standard score.
        :param recompute_normalization: If True, compute the normalization coefficients.
        """

        self.__name: str = self.__class__.__name__

        # Check directory variable
        if existing_dir is not None:
            if not isdir(existing_dir):
                raise ValueError(f"[{self.__name}] The given 'existing_dir'={existing_dir} does not exist.")
            if len(existing_dir.split(sep)) > 1 and existing_dir.split(sep)[-1] == 'dataset':
                existing_dir = join(*existing_dir.split(sep)[:-1])
            if not isabs(existing_dir):
                existing_dir = abspath(existing_dir)

        # Check storage variables
        if mode is not None:
            if mode.lower() not in (available_modes := ['training', 'validation', 'prediction']):
                raise ValueError(f"[{self.__name}] The given 'mode'={mode} must be in {available_modes}.")

        # DatabaseManager parameterization
        self.existing_dir: Optional[str] = existing_dir
        self.mode: Optional[str] = mode
        self.shuffle: bool = shuffle
        self.normalize: bool = normalize
        self.recompute_normalization: bool = recompute_normalization

    def __str__(self):

        description = "\n"
        description += f"{self.__name}\n"
        description += f"    Existing directory: {False if self.existing_dir is None else self.existing_dir}\n"
        description += f"    Mode: {self.mode}\n"
        description += f"    Shuffle: {self.shuffle}\n"
        description += f"    Normalize: {self.normalize}\n"
        description += f"    Recompute normalization: {self.recompute_normalization}\n"
        return description
