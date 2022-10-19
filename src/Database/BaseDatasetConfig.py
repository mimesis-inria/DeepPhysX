from typing import Optional
from os.path import isdir, sep, join


class BaseDatasetConfig:

    def __init__(self,
                 existing_dir: Optional[str] = None,
                 mode: Optional[str] = None,
                 max_file_size: Optional[float] = None,
                 shuffle: bool = True,
                 normalize: bool = True,
                 recompute_normalization: bool = False):
        """
        Configuration class to parameterize the Dataset and its Manager.

        :param existing_dir: Path to an existing Dataset repository.
        :param mode: Specify the Dataset mode that should be used between 'training', 'validation' and 'running'.
        :param max_file_size: Maximum size (in Gb) of a single dataset file.
        :param shuffle: Specify if the Dataset should be shuffled when a batch is taken.
        :param normalize: If True, the data will be normalized using standard score.
        :param recompute_normalization: If True, compute the normalization coefficients.
        """

        self.name: str = self.__class__.__name__

        # Check directory variable
        if existing_dir is not None:
            if type(existing_dir) != str:
                raise TypeError(f"[{self.name}] The given 'existing_dir'={existing_dir} must be a str.")
            if not isdir(existing_dir):
                raise ValueError(f"[{self.name}] The given 'existing_dir'={existing_dir} does not exist.")
            if len(existing_dir.split(sep)) > 1 and existing_dir.split(sep)[-1] == 'dataset':
                existing_dir = join(*existing_dir.split(sep)[:-1])

        # Check storage variables
        if mode is not None:
            if type(mode) != str:
                raise TypeError(f"[{self.name}] The given 'mode'={mode} must be a str.")
            if mode.lower() not in (available_modes := ['training', 'validation', 'running']):
                raise ValueError(f"[{self.name}] The given 'mode'={mode} must be in {available_modes}.")
        if max_file_size is not None:
            if type(max_file_size) not in [int, float]:
                raise TypeError(f"[{self.name}] The given 'max_file_size'={max_file_size} must be a float.")
            max_file_size = int(max_file_size * 1e9) if max_file_size > 0 else None
        if type(shuffle) != bool:
            raise TypeError(f"[{self.name}] The given 'shuffle'={shuffle} must be a bool.")
        if type(normalize) != bool:
            raise TypeError(f"[{self.name}] The given 'normalize'={normalize} must be a bool.")
        if type(recompute_normalization) != bool:
            raise TypeError(f"[{self.name}] The given 'recompute_normalization'={recompute_normalization} must be a "
                            f"bool.")

        # DatasetManager parameterization
        self.existing_dir: Optional[str] = existing_dir
        self.mode: Optional[str] = mode
        self.max_file_size: int = max_file_size
        self.shuffle: bool = shuffle
        self.normalize: bool = normalize
        self.recompute_normalization: bool = recompute_normalization

    def __str__(self) -> str:

        description = "\n"
        description += f"{self.name}\n"
        description += f"    Existing directory: {False if self.existing_dir is None else self.existing_dir}\n"
        description += f"    Mode: {self.mode}\n"
        description += f"    Max size: {self.max_file_size}\n"
        description += f"    Shuffle: {self.shuffle}\n"
        description += f"    Normalize: {self.normalize}\n"
        description += f"    Recompute normalization: {self.recompute_normalization}\n"
        return description