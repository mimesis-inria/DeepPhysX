from typing import Dict, List, Optional, Tuple
from numpy import array, ndarray, concatenate, save, arange
from numpy.random import shuffle
from collections import namedtuple


class BaseDataset:
    """
    | BaseDataset is a dataset class to store any data from a BaseEnvironment or from files.
    | Given data is split into input data and output data.
    | Saving data results in multiple partitions of input and output data.

    :param namedtuple config: Namedtuple which contains BaseDataset parameters
    """

    def __init__(self, config: namedtuple):

        self.name: str = self.__class__.__name__

        # Data fields containers
        self.data_type = ndarray
        self.fields: List[str] = ['input', 'output']
        self.data: Dict[str, ndarray] = {'input': array([]), 'output': array([])}
        self.shape: Dict[str, Optional[List[int]]] = {'input': None, 'output': None}

        # Indexing
        self.shuffle_pattern: Optional[List[int]] = None
        self.current_sample: int = 0

        # Dataset memory
        self.max_size: int = config.max_file_size
        self.batch_per_field: Dict[str, int] = {field: 0 for field in ['input', 'output']}
        self.__empty: bool = True

    @property
    def nb_samples(self) -> int:
        """
        | Property returning the current number of samples

        :return: The current number of samples in all partitions
        """

        return max([len(self.data[field]) for field in self.fields])

    def is_empty(self) -> bool:
        """
        | Check if the fields of the dataset are empty. A field is considered as non-empty if it is filled with another
          sample.

        :return: The Dataset is empty or not
        """

        # The empty flag is set to False once the Dataset is considered as non-empty
        if not self.__empty:
            return False
        # Check each registered data field
        for field in self.fields:
            # Dataset is considered as non-empty if a field is filled with another sample
            if self.batch_per_field[field] > 1:
                self.__empty = False
                return False
        # If all field are considered as non-empty then the Dataset is empty
        return True

    def init_data_size(self, field: str, shape: List[int]) -> None:
        """
        | Store the original shape of data. Reshape data containers.

        :param str field: Data field name
        :param List[int] shape: Shape of the corresponding tensor
        """

        # Store the original data shape
        self.shape[field] = shape
        # Reshape the data container
        self.data[field] = array([]).reshape((0, *shape))

    def get_data_shape(self, field: str) -> List[int]:
        """
        | Returns the data shape of field.

        :param str field: Data field name
        :return: Data shape for field
        """

        return self.shape[field]

    def init_additional_field(self, field: str, shape: List[int]) -> None:
        """
        | Register a new data field.

        :param str field: Name of the data field
        :param List[int] shape: Data shape
        """

        # Register the data field
        self.fields.append(field)
        # Init the number of adds in the field
        self.batch_per_field[field] = 0
        # Init the field shape
        self.init_data_size(field, shape)

    def empty(self) -> None:
        """
        | Empty the dataset.
        """

        # Reinit each data container
        for field in self.fields:
            self.data[field] = array([]) if self.shape[field] is None else array([]).reshape((0, *self.shape[field]))
        # Reinit indexing variables
        self.shuffle_pattern = None
        self.current_sample = 0
        self.batch_per_field = {field: 0 for field in self.fields}
        self.__empty = True

    def memory_size(self, field: Optional[str] = None) -> int:
        """
        | Return the actual memory size of the dataset if field is None. Otherwise, return the actual memory size of the
          field.

        :param Optional[str] field: Name of the data field
        :return: Size in bytes of the current dataset.
        """

        # Return the total memory size
        if field is None:
            return sum([self.data[field].nbytes for field in self.fields])
        # Return the memory size for the specified field
        return self.data[field].nbytes

    def check_data(self, field: str, data: ndarray) -> None:
        """
        | Check if the data is a numpy array.

        :param str field: Values at 'input' or anything else. Define if the associated shape is correspond to input
                          shape or output one.
        :param ndarray data: New data
        """

        if type(data) != self.data_type:
            raise TypeError(f"[{self.name}] Wrong data type in field '{field}': numpy array required, got {type(data)}")

    def add(self, field: str, data: ndarray, partition_file: Optional[str] = None) -> None:
        """
        | Add data to the dataset.

        :param str field: Name of the data field
        :param ndarray data: New data as batch of samples
        :param Optional[str] partition_file: Path to the file in which to write the data
        """

        # Check data type
        self.check_data(field, data)

        # Check if field is registered
        if field not in self.fields:
            # Fields can be register only if Dataset is empty
            if not self.is_empty():
                raise ValueError(f"[{self.name}] A new field {field} tries to be created as Dataset is non empty. This "
                                 f"will lead to a different number of sample for each field of the dataset.")
            # Add new field if not registered
            self.init_additional_field(field, data[0].shape)

        # Check data size initialization
        if self.shape[field] is None:
            self.init_data_size(field, data[0].shape)

        # Add batched samples
        self.data[field] = concatenate((self.data[field], data))
        # Save in partition
        if partition_file is not None:
            self.save(field, partition_file)

        # Update sample indexing in dataset
        self.batch_per_field[field] += 1
        self.current_sample = max([len(self.data[f]) for f in self.fields])

    def save(self, field: str, file: str) -> None:
        """
        | Save the corresponding field of the Dataset.

        :param str field: Name of the data field
        :param str file: Path to the file in which to write the data
        """

        save(file, self.data[field])

    def set(self, field: str, data: ndarray) -> None:
        """
        | Set a full field of the dataset.

        :param str field: Name of the data field
        :param ndarray data: New data as batch of samples
        """

        # Check data type
        self.check_data(field, data)

        # Check if field is registered
        if field not in self.fields:
            # Add new field if not registered
            self.init_additional_field(field, data[0].shape)

        # Check data size initialization
        if self.shape[field] is None:
            self.init_data_size(field, data[0].shape)

        # Set the full field
        self.data[field] = data

        # Update sample indexing in dataset
        self.__empty = False
        self.current_sample = 0

    def get(self, field: str, idx_begin: int, idx_end: int) -> ndarray:
        """
        | Get a batch of data in 'field' container.

        :param str field: Name of the data field
        :param int idx_begin: Index of the first sample
        :param int idx_end: Index of the last sample
        :return: Batch of data from 'field'
        """

        indices = slice(idx_begin, idx_end) if self.shuffle_pattern is None else self.shuffle_pattern[idx_begin:idx_end]
        return self.data[field][indices]

    def shuffle(self) -> None:
        """
        | Define a random shuffle pattern.
        """

        # Nothing to shuffle if Dataset is empty
        if self.is_empty():
            return
        # Generate a shuffle pattern
        self.shuffle_pattern = arange(self.nb_samples)
        shuffle(self.shuffle_pattern)

    def __str__(self) -> str:
        """
        :return: String containing information about the BaseDatasetConfig object
        """

        description = "\n"
        description += f"  {self.name}\n"
        description += f"    Max size: {self.max_size}\n"
        description += f"    Data fields: {self.fields}"
        for field in self.fields:
            description += f"      {field} shape: {self.shape[field]}"
        return description
