from typing import Optional, Tuple, Self
from threading import Thread
from numpy import ndarray, array
from numpy.linalg import norm
from vedo import Mesh, show


class VisiblePoints:

    def __init__(self, mesh_positions: ndarray, mesh_cells: ndarray):
        """
        Extract the visible vertices of a mesh depending on a camera parameters.

        :param mesh_positions: Positions array of the mesh.
        :param mesh_cells: Cells array of the mesh.
        """

        # Mesh information
        self.__mesh = Mesh([mesh_positions, mesh_cells]).compute_normals()

        # Camera information
        self.__camera = {'pos': array([0., 0., 1.]), 'focalPoint': array([0., 0., 0.]), 'viewAngle': 30.}
        self.__distance_threshold = float('Inf')
        self.__result = None

    def set_camera(self,
                   camera_position: Optional[ndarray] = None,
                   focal_point: Optional[ndarray] = None,
                   view_angle: Optional[float] = None,
                   distance_threshold: Optional[float] = None) -> Self:
        """
        Define the camera parameters.

        :param camera_position: Current position of the camera.
        :param focal_point: Current focal point of the camera.
        :param view_angle: Current view angle of the camera.
        :param distance_threshold: Max distance to the camera for a point to be detected.
        """

        self.__camera = {'pos': camera_position if camera_position is not None else self.__camera['pos'],
                         'focalPoint': focal_point if focal_point is not None else self.__camera['focalPoint'],
                         'viewAngle': view_angle if view_angle is not None else self.__camera['viewAngle']}
        self.__distance_threshold = self.__distance_threshold if distance_threshold is None else distance_threshold
        return self

    def extract(self, new_positions: Optional[ndarray] = None) -> Tuple[ndarray, ndarray]:
        """
        Return the positions and the indices of the current visible nodes.

        :param new_positions: Current positions of the mesh.
        :return: Tuple of arrays containing visible_points_positions and visible_points_indices.
        """

        thread = Thread(target=self.__extract, args=(new_positions,))
        thread.start()
        thread.join()
        return self.res


    def __extract(self, new_positions: Optional[ndarray] = None):

        # Create the mesh
        if new_positions is not None:
            self.__mesh.vertices = new_positions
            self.__mesh.compute_normals()

        # Extract visible nodes
        plt = show(self.__mesh, new=True, camera=self.__camera, offscreen=True)
        visible_pcd = self.__mesh.visible_points()
        # plt.close()
        # del plt
        # plt = show(self.__mesh, visible_pcd, new=True, camera=self.__camera, offscreen=False)
        # plt.close()
        # del plt

        # Build visible node indices and position list filtered with distance to camera
        filtered_visible_points = []
        filtered_visible_idx = []
        for point in visible_pcd.vertices:
            if norm(point - self.__camera['pos']) < self.__distance_threshold:
                filtered_visible_points.append(point)
                filtered_visible_idx.append(self.__mesh.closest_point(pt=point, return_point_id=True))
        self.res = array(filtered_visible_points), array(filtered_visible_idx)
