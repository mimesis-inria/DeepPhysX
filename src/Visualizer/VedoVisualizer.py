from typing import List, Any, Dict, Union
from os.path import join as osPathJoin
from os import makedirs
from vedo import Plotter, Text2D, show

from DeepPhysX.Core.Visualizer.VedoObjects import VedoObjects, ObjectDescription

Viewers = Dict[int, Dict[str, Union[str, List[Any], bool, Plotter]]]


class VedoVisualizer:
    """
    | Visualizer class to display VisualInstances in a 2D/3D environment.
    | VedoVisualiser use the vedo library to display 3D models.
    | Objects are given in the init_view function.
    | Updates to these objects are achieved by using update_visualizer and update_instances functions.
    """

    def __init__(self):

        self.scene: Dict[int, VedoObjects] = {}
        self.default_viewer_id: int = 9
        self.viewers: Viewers = {self.default_viewer_id: {'title': f"Vedo_axes_{self.default_viewer_id}",
                                                          'instances': [],
                                                          'sharecam': True,
                                                          'interactive': True}}
        self.objects_rendered_in: Dict[str, (int, int)] = {}
        self.info = Text2D("Press 'q' to\nstart session")

        # Wrong samples parameters
        self.folder_path: str = ''
        self.nb_saved: int = 0
        self.nb_screenshots: int = 0

    def init_view(self, data_dict: Dict[int, Dict[int, Dict[str, Union[Dict[str, Any], Any]]]]) -> None:
        """
        | Initialize VedoVisualizer class by parsing the scenes hierarchy and creating VisualInstances.
        | OBJECT DESCRIPTION DICTIONARY is usually obtained using the corresponding factory (VedoObjectFactory).
        | data_dict example:
        |       {SCENE_1_ID: {OBJECT_1.1_ID: {CONTENT OF OBJECT_1.1 DESCRIPTION DICTIONARY},
        |                     ...
        |                     OBJECT_1.N_ID: {CONTENT OF OBJECT_1.N DESCRIPTION DICTIONARY}
        |                     },
        |        ...
        |        SCENE_M_ID: {OBJECT_M.1_ID: {CONTENT OF OBJECT_K.1 DESCRIPTION DICTIONARY},
        |                     ...
        |                     OBJECT_M.K_ID: {CONTENT OF OBJECT_K.P DESCRIPTION DICTIONARY}
        |                    }
        |        }

        :param data_dict: Dictionary describing the scene hierarchy and object parameters
        :type data_dict: Dict[int, Dict[int, Dict[str, Union[Dict[str, Any], Any]]]]
        """

        # For each scene/client
        for scene_id in sorted(data_dict):

            # Add a VedoObjects container for the scene
            self.scene.update({scene_id: VedoObjects()})
            scene = self.scene[scene_id]

            # Create each object in current scene/client
            for object_id in data_dict[scene_id]:
                scene.create_object(data_dict[scene_id][object_id])
            objects_dict = scene.objects_factory.objects_dict

            # Deal with all the windows and object attached to these windows first
            remaining_object_id = set(objects_dict.keys())
            for window_id in scene.objects_factory.windows_id:

                # Removes the window we are dealing with from the object we have to add to the plotter
                remaining_object_id -= {window_id}
                # Vedo can only handle 1 axe type per viewer, so we create as many viewers as needed
                viewer_id = objects_dict[window_id]['axes']

                # Existing viewer: ensure all objects share the same window parameters
                if viewer_id in self.viewers:
                    # If at least one need a shared camera then we set True for all
                    self.viewers[viewer_id]['sharecam'] |= objects_dict[window_id]['sharecam']
                    # If one requires that the window is not interactive then it's not interactive for all
                    self.viewers[viewer_id]['interactive'] &= objects_dict[window_id]['interactive']

                # New viewer: init parameters
                else:
                    self.viewers[viewer_id] = {'sharecam': objects_dict[window_id]['sharecam'],
                                               'interactive': objects_dict[window_id]['interactive'],
                                               'instances': [],
                                               'title': f"Vedo_axes_{objects_dict[window_id]['axes']}"}

                # Add the objects to the corresponding list
                for object_id in objects_dict[window_id]['objects_id']:
                    # Affects the object in the existing window
                    if -1 < objects_dict[object_id]['at'] < len(self.viewers[viewer_id]['instances']):
                        self.viewers[viewer_id]['instances'][objects_dict[object_id]['at']].append([scene_id,
                                                                                                    object_id])
                    # Affects the object in the next non-existing window
                    else:
                        objects_dict[object_id]['at'] = len(self.viewers[viewer_id]['instances'])
                        self.viewers[viewer_id]['instances'].append([[scene_id, object_id]])

                # Remove all the objects attached to the window from the object to deal with
                remaining_object_id -= set(objects_dict[window_id]['objects_id'])

            # Deals with the remaining objects that are not specified in windows
            for object_id in remaining_object_id:

                # Affects the object in the existing window
                if -1 < objects_dict[object_id]['at'] < len(self.viewers[self.default_viewer_id]['instances']):
                    self.viewers[self.default_viewer_id]['instances'][objects_dict[object_id]['at']].append([scene_id,
                                                                                                             object_id])

                # Affects the object in the next non-existing window
                else:
                    objects_dict[object_id]['at'] = len(self.viewers[self.default_viewer_id]['instances'])
                    self.viewers[self.default_viewer_id]['instances'].append([[scene_id, object_id]])

        # Once all objects are created we create the plotter with the corresponding parameters
        for viewer_id in list(self.viewers.keys()):

            # If no objects created for the viewer, remove it
            if len(self.viewers[viewer_id]['instances']) == 0:
                del self.viewers[viewer_id]
                continue

            # # Create plotter
            # self.viewers[viewer_id]['plotter'] = Plotter(N=len(self.viewers[viewer_id]['instances']),
            #                                              title=self.viewers[viewer_id]['title'],
            #                                              axes=viewer_id,
            #                                              sharecam=self.viewers[viewer_id]['sharecam'],
            #                                              interactive=self.viewers[viewer_id]['interactive'])
            # self.viewers[viewer_id]['plotter'].add(self.info, at=0)
            #
            # # self.viewers[viewer_id]['instances'] is a list containing lists of instances
            # # Each sublist contains all instances present in a window hence, each sublist has it own "at"
            # for at, ids in enumerate(self.viewers[viewer_id]['instances']):
            #     for scene_id, object_in_scene_id in ids:
            #         # Add object instance in the actors list of plotter
            #         self.viewers[viewer_id]['plotter'].add(
            #             self.scene[scene_id].objects_instance[object_in_scene_id], at=at, render=False)
            #         # Register the object rendering location
            #         self.objects_rendered_in[f'{scene_id}_{object_in_scene_id}'] = (viewer_id, at)
            #
            # # Render viewer
            # self.viewers[viewer_id]['plotter'].show(interactive=True)
            # self.viewers[viewer_id]['plotter'].remove(self.info)

            actors = []
            for at, ids in enumerate(self.viewers[viewer_id]['instances']):
                actors.append([])
                if at == 0:
                    actors[-1].append(self.info)
                for scene_id, object_in_scene_id in ids:
                    actors[-1].append(self.scene[scene_id].objects_instance[object_in_scene_id])
                    self.objects_rendered_in[f'{scene_id}_{object_in_scene_id}'] = (viewer_id, at)
            self.viewers[viewer_id]['plotter'] = show(actors,
                                                      N=len(actors),
                                                      title=self.viewers[viewer_id]['title'],
                                                      axes=viewer_id,
                                                      sharecam=self.viewers[viewer_id]['sharecam'],
                                                      interactive=self.viewers[viewer_id]['interactive'])
            self.viewers[viewer_id]['plotter'].remove(self.info)

    def render(self) -> None:
        """
        | Call render on all valid plotter.
        """

        # Update all objects
        self.update_instances()
        # Render all plotters
        for viewer_id in self.viewers:
            self.viewers[viewer_id]['plotter'].render()
            self.viewers[viewer_id]['plotter'].allowInteraction()

    def update_instances(self) -> None:
        """
        | Call update_instance on all updates object description
        """

        # Update in every scene/client
        for scene_id in self.scene:
            # Update every object of the current scene
            for object_id in self.scene[scene_id].objects_instance:
                # Get the rendering location of the object
                viewer_data = self.objects_rendered_in[f'{scene_id}_{object_id}']
                plotter = self.viewers[viewer_data[0]]['plotter']
                at = viewer_data[1]
                # _, self.viewers[viewer_data[0]]['plotter'] = self.scene[scene_id].update_instance(
                #     object_id, (self.viewers[viewer_data[0]]['plotter'], viewer_data[1]))
                self.scene[scene_id].update_instance(object_id, (plotter, at))

    def update_visualizer(self, data_dict: Dict[int, Dict[int, ObjectDescription]]) -> None:
        """
        | Call update_object_dict on all designed objects.

        :param data_dict: Dictionary describing the scene hierarchy and object parameters
        :type data_dict: Dict[int, Dict[int, Dict[str, Union[Dict[str, Any], Any]]]]
        """

        for scene_id in data_dict:
            for object_id in data_dict[scene_id]:
                self.scene[scene_id].update_object(object_id, data_dict[scene_id][object_id])

    def save_sample(self, session_dir: str, viewer_id: int) -> None:
        """
        Save the samples as a .npz file.

        :param str session_dir: Directory in which to save the file
        :param int viewer_id: id of the designed viewer
        """

        if self.folder_path == "":
            self.folder_path = osPathJoin(session_dir, 'dataset', 'wrong_samples')
            makedirs(self.folder_path)
            from DeepPhysX.Core.Utils import wrong_samples
            import shutil
            shutil.copy(wrong_samples.__file__, self.folder_path)
        filename = osPathJoin(self.folder_path, f'wrong_sample_{self.nb_saved}.npz')
        self.nb_saved += 1
        self.viewers[viewer_id]['plotter'].export(filename=filename)

    def save_screenshot(self, session_dir: str) -> None:
        """
        | Save a screenshot of each viewer in the dataset folder_path of the session.

        :param str session_dir: Directory in which to save the file
        """

        # Check folder_path existence
        if self.folder_path == "":
            self.folder_path = osPathJoin(session_dir, 'dataset', 'samples')
            makedirs(self.folder_path)

        # Save a screenshot for each viewer
        for viewer_id in self.viewers.keys():
            filename = osPathJoin(self.folder_path, f'screenshot_{self.nb_screenshots}.png')
            self.nb_screenshots += 1
            self.viewers[viewer_id]['plotter'].screenshot(filename=filename)
