import copy
import yaml
import importlib


def BaseYamlExporter(filename: str=None, var_dict:dict=None):
    """
    | Exports variables in a yaml file, excluding classes, modules and functions. Additionally, variables with a name in
    | excluded will not be exported.
    :param str filename: Path to the file in which var_dict will be saved after filtering
    :param dict var_dict: Dictionnary containing the key:val pairs to be saved. Key is a variable name and val its value
    """
    export_dict = copy.deepcopy(var_dict)
    def convert_repr_to_name(repr_str: str):
        if repr_str.__contains__("<class "): #Class object, not instanciated
            return repr_str.split("<class '")[1].split("'>")[0]
        else:
            raise ValueError(f"BaseYamlExporter: {repr_str} could not be converted to an object name.")
    def convert_variables(var_container):
        var_container_type = type(var_container)
        if isinstance(var_container, dict):
            keys = list(var_container.keys())
            if 'excluded' in keys: #Special keyword that specify which keys should be removed
                for exclude_key in var_container['excluded']:
                    if exclude_key in var_container: var_container.pop(exclude_key) #Remove the key listed in excluded
                    keys = list(var_container.keys()) #Update the keys
        elif isinstance(var_container, (tuple, list, set)): #Is not a dict but is iterable.
            keys = range(len(var_container))
            var_container = list(var_container) #Allows to change elements in var_container
        else:
            raise ValueError(f"BaseYamlExporter: encountered an object to convert which is not a dict, tuple or list.")
        for k in keys:
            v = var_container[k]
            if isinstance(v, type):  # Object is just a type, not an instance
                new_val = convert_repr_to_name(repr(v))
                new_val = dict(type=new_val)
            elif hasattr(v, '__iter__') and not isinstance(v, str):  # Object contains other objects
                new_val = convert_variables(v)
            else:  # Object is assumed to not contain other objects
                new_val = v
            var_container[k] = new_val
        if var_container_type in (tuple, set):
            var_container = var_container_type(var_container) #Convert back to original type
        return var_container

    convert_variables(export_dict)
    if filename is not None:
        #Export to yaml file
        with open(filename,'w') as f:
            print(f"[BaseYamlExporter] Saving conf to {filename}.")
            yaml.dump(export_dict, f)
    return export_dict


def BaseYamlLoader(filename: str):
    with open(filename, 'r') as f:
        loaded_dict = yaml.load(f, yaml.Loader)
    convert_variables(loaded_dict)
    return loaded_dict


def convert_variables(var_container):
    var_container_type = type(var_container)
    if isinstance(var_container, dict):
        keys = list(var_container.keys())
    elif isinstance(var_container, (tuple, list, set)):  # Is not a dict but is iterable.
        keys = range(len(var_container))
        var_container = list(var_container) #Allows to change elements in var_container
    else:
        raise ValueError(f"BaseYamlLoader: encountered an object to convert which is not a dict, tuple or list.")
    for k in keys:
        v = var_container[k]
        # Detection of a type object that was converted to str
        if isinstance(v,dict) and len(v) == 1 and 'type' in v and isinstance(v['type'], str):
            new_val = convert_type_str_to_type(v['type'])
        elif hasattr(v, '__iter__') and not isinstance(v, str):  # Object contains other objects
            new_val = convert_variables(v)
        else:
            new_val = v
        var_container[k] = new_val
    if var_container_type in (tuple, set):
        var_container = var_container_type(var_container) #Convert back to original type
    return var_container


def convert_type_str_to_type(name: str):
        module = importlib.import_module('.'.join(name.split('.')[:-1]))
        object_name_in_module = name.split('.')[-1]
        return getattr(module, object_name_in_module)