from vedo import Mesh
from numpy import array

from DeepPhysX_Core.Visualizer.VedoObjectFactories.BaseObjectFactory import BaseObjectFactory, parse_wrapper, \
    update_wrapper, ObjectDescription


class MeshFactory(BaseObjectFactory):
    """
    | MeshFactory is a class that represent the data of a Mesh visual object.
    | MeshFactory defines the parse and update procedures of Mesh object according to Vedo.
    """

    def __init__(self):

        BaseObjectFactory.__init__(self)

        self.type = 'Mesh'
        self.grammar_plug = ['positions', 'cells', 'computeNormals', 'wireframe']
        self.grammar.extend(self.grammar_plug)
        self.default_values.update({self.grammar_plug[0]: None,
                                    self.grammar_plug[1]: None,
                                    self.grammar_plug[2]: False,
                                    self.grammar_plug[3]: False})
        self.number_of_dimensions = 0

    @parse_wrapper()
    def parse(self, data_dict: ObjectDescription) -> None:
        """
        | Parse the given dictionary and fill the parsed_data member accordingly.
        | Note: It is the wrapper that return the parsed_data.

        :param data_dict: Dictionary to parse
        :type data_dict: Dict[str, Union[Any, Dict[str, Any]]]
        """

        # Parse 'positions' field
        if 'position' in data_dict:
            data_dict[self.grammar_plug[0]] = data_dict.pop('position')
        if self.grammar_plug[0] in data_dict:
            self.dirty_fields.append(self.grammar_plug[0])
            self.parsed_data[self.grammar_plug[0]] = self.parse_vector(data_dict[self.grammar_plug[0]])

        # Parse 'cells' field
        if 'cell' in data_dict:
            data_dict[self.grammar_plug[1]] = data_dict.pop('cells')
        if self.grammar_plug[1] in data_dict:
            self.dirty_fields.append(self.grammar_plug[1])
            self.parsed_data[self.grammar_plug[1]] = array(data_dict['cells'], dtype=int)

    @update_wrapper()
    def update_instance(self, instance: Mesh) -> Mesh:
        """
        | Update the given VisualInstance instance.

        :param Mesh instance: Vedo object to update with its current parsed_data values
        :return: The updated VisualInstance
        """

        # Update positions
        if self.grammar_plug[0] in self.dirty_fields:
            instance.points(self.parsed_data[self.grammar_plug[0]])
            self.dirty_fields.remove(self.grammar_plug[0])

        return instance
