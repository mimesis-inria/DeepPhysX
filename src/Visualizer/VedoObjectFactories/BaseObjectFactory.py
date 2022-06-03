from typing import Union, Optional, Dict, Any, Callable, TypeVar, List
from vedo import utils, Mesh, Glyph, Marker, Points, Arrows
from numpy import array, ndarray, stack, zeros, c_

VisualInstance = Union[Mesh, Glyph, Marker, Points, Arrows]
ObjectDescription = Dict[str, Union[Any, Dict[str, Any]]]
Vector = Union[ndarray, List[ndarray], list, List[list]]

ParseParameters = TypeVar("ParseParameters", Callable[..., Any], Any)  # ParamSpec('ParseParameters')
UpdateParameters = TypeVar("UpdateParameters", Callable[..., Any], Any)  # ParamSpec('UpdateParameters')


class parse_wrapper:
    """
    | Class wrapper. Wraps the parse function of all vedo BaseObjectFactory subclasses.
    | Allows to only define the specialized parse procedure in the subclasses while still running the general parse
      procedure.
    | When calling the parse function, the caller is passed as args[0] (self) and the arguments are either in args or
      kwargs.
    | This class only implements __call__.
    """

    def __call__(self, specialized_parse: Callable[[ParseParameters], None]) -> Callable[[Any, Any], ObjectDescription]:
        """
        | Wrap the parse call.
        | Parses all the attributes shared between subclasses of BaseObjectFactory.

        :param specialized_parse: parse function of a subclass of BaseObjectFactory
        :return: general_update: The general parse function
        """

        def general_parse(*args: Any, **kwargs: Any) -> ObjectDescription:
            """
            | Run the general parse procedure, then run the one defined in the calling subclass of BaseObjectFactory.

            :param Tuple[CallingSubClass,...] args: List with the calling subclass and maybe the dictionary of data to
                                                    be parsed
            :param Dict[str, Any] kwargs: "Any" here will be the dictionary of data to be parsed if not present in args
            :return: The fully parsed and evaluated dictionary
            """

            # Parse only takes 2 arguments (self and data_dict)
            if len(args) + len(kwargs) != 2:
                return {'': None}

            object_factory = args[0]
            if len(args) == 2:
                data_dict = args[1]
            else:
                (_, data_dict), *rest = kwargs.items()  # : (Any, ObjectDescription, Any)

            # Load all parameters directly given with the appropriate name
            object_factory.parsed_data.update({word: data_dict[word] for word in object_factory.grammar
                                               if word in data_dict})
            object_factory.parsed_data['type'] = object_factory.type

            # Run the specialized parse code
            specialized_parse(self=object_factory, data_dict=data_dict)

            # Default init of all not given parameters
            for word in object_factory.grammar:
                if word not in object_factory.parsed_data:
                    object_factory.parsed_data[word] = object_factory.default_values[word]

            return object_factory.parsed_data

        return general_parse


class update_wrapper:
    """
    | Class wrapper. Wraps the update_instance function of all vedo BaseObjectFactory subclasses.
    | Allows to only define the specialized update_instance procedure in the subclasses while still running the general
      update_instance procedure.
    | When calling the update_instance function, the caller is passed as args[0] (self) and the arguments are either in
      args or kwargs.
    | This class only implements __call__.
    """

    def __call__(self, specialized_update: Callable[[UpdateParameters], None]) -> Callable[[Any, Any], VisualInstance]:
        """
        | Wrap the update_instance call.
        | Updates all the attributes shared between subclasses of BaseObjectFactory.

        :param specialized_update: update_instance function of a subclass of BaseObjectFactory
        :return: general_update: The general update function
        """

        def general_update(*args: Any, **kwargs: Any) -> VisualInstance:
            """
            | Run the general update_instance procedure, then run the one defined in the calling subclass of
              BaseObjectFactory.

            :param Tuple[CallingSubClass,...] args: List with the calling subclass and the instance to update
            :param Dict[str, Any] kwargs: "Any" here will be the instance to be updated if not present in args
            :return: The updated VisualObject
            """

            # Parse only takes 2 arguments (self and instance)
            if len(args) + len(kwargs) != 2:
                return

            # First object is always self, it is passed implicitly
            object_factory = args[0]

            # Either in kwargs or 2nd argument in args but always last with update_instance definition
            instance = kwargs['instance'] if 'instance' in kwargs else args[-1]

            # Update 'colormap' if dirty
            if 'colormap' in object_factory.dirty_fields:
                instance.c(object_factory.parsed_data['colormap'])
                object_factory.dirty_fields.remove('colormap')

            # Update 'scalar_field' if dirty
            if 'scalar_field' in object_factory.dirty_fields:
                instance.addPointArray(input_array=object_factory.parsed_data['scalar_field'],
                                       name=object_factory.parsed_data['scalar_field_name'])
                object_factory.dirty_fields.remove('scalar_field')
                object_factory.dirty_fields.remove('scalar_field_name')

            # Update 'alpha' if dirty
            if 'alpha' in object_factory.dirty_fields:
                instance.alpha(object_factory.parsed_data['alpha'])
                object_factory.dirty_fields.remove('alpha')

            # Update 'color' if dirty
            if 'c' in object_factory.dirty_fields:
                instance.c(object_factory.parsed_data['c'])
                object_factory.dirty_fields.remove('c')

            # Run the specialized update code
            return specialized_update(self=object_factory, instance=instance)

        return general_update


class BaseObjectFactory:
    """
    | Base class of all the Visual object visualizer.
    | BaseObjectFactory defines the parse and update procedures of all the object visualizer.
    """

    def __init__(self):

        self.type: str = ""
        self.grammar: List[str] = ['c', 'alpha', 'at', 'colormap', 'scalar_field', "scalar_field_name"]
        self.grammar_plug: List[str] = []
        self.default_values: ObjectDescription = {self.grammar[0]: 'b', self.grammar[1]: 1.0, self.grammar[2]: -1,
                                                  self.grammar[3]: 'jet', self.grammar[4]: [],
                                                  self.grammar[5]: "scalar_field"}
        self.parsed_data: ObjectDescription = {}
        self.dirty_fields: List[str] = []

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        """
        | Parse the given dictionary and fill the parsed_data member accordingly.
        | Note: It is the wrapper that return the parsed_data.

        :param data_dict: Dictionary to parse
        :type data_dict: Dict[str, Union[Any, Dict[str, Any]]]
        """

        raise NotImplementedError

    @update_wrapper()
    def update_instance(self, instance: VisualInstance) -> VisualInstance:
        """
        | Update the given VisualInstance instance.

        :param instance: Vedo object to update with its current parsed_data values
        :type instance: Union[Mesh, Glyph, Marker, Points, Arrows]
        :return: The updated VisualInstance
        """

        raise NotImplementedError

    def get_data(self) -> ObjectDescription:
        """
        | Get the visualization data description.

        :return: A Dict[str, Any] that represent the parsed_data member
        """

        return self.parsed_data

    @staticmethod
    def parse_vector(vec: Optional[Vector], wrap: bool = True) -> Optional[ndarray]:
        """
        | Helper function (static method) that parses a vector field.

        :param Optional[numpy.ndarray, List[numpy.ndarray]] vec: Vector to parse
        :param bool wrap: When True add a dimension to the vector (ex: if the input data shape is [N, 3], data will
                          have shape [1, N, 3]
        :return: A numpy.ndarray that contains the parsed vector and an additional dimension if wrap is True
        """

        if utils.isSequence(vec):
            if not utils.isSequence(vec[0]) and wrap:
                vec = [vec]
            vec = array(vec)
            n = len(vec)
            # Assume vector is in the format [all_x, all_y, all_z]
            if n == 3:
                if utils.isSequence(vec[0]) and len(vec[0]) > 3:
                    vec = stack((vec[0], vec[1], vec[2]), axis=1)
            # Assume vector is in the format [all_x, all_y, 0]
            if n == 2:
                if utils.isSequence(vec[0]) and len(vec[0]) > 3:
                    vec = stack((vec[0], vec[1], zeros(len(vec[0]))), axis=1)
                else:
                    vec = array([vec[0], vec[1], 0])
            # Make it 3D
            if n and wrap and len(vec[0]) == 2:
                vec = c_[array(vec), zeros(len(vec))]

        return vec
