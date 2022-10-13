from typing import Any
from collections import namedtuple


def make_config(configuration_object: Any,
                configuration_name: str,
                **kwargs) -> namedtuple:
    """
    Create a namedtuple which gathers all the parameters for any configuration object.
    For a child config class, only new items are required since parent's items will be added by default.

    :param configuration_object: Instance of any Config class.
    :param configuration_name: Name of the variable containing the namedtuple.
    :param kwargs: Parameters to add to the namedtuple.
    :return: Namedtuple which contains newly added parameters.
    """

    # Get items set as keyword arguments
    fields = tuple(kwargs.keys())
    args = tuple(kwargs.values())

    # Check if a dataset_config already exists (child class will have the parent's config by default)
    if configuration_name in configuration_object.__dict__:
        configuration: namedtuple = configuration_object.__getattr__(configuration_name)
        # Get items set in the existing config
        for key, value in configuration._asdict().items():
            # Only new items are required for children, check if the parent's items are set again anyway
            if key not in fields:
                fields += (key,)
                args += (value,)

    # Create namedtuple with collected items
    return namedtuple(configuration_name, fields)._make(args)
