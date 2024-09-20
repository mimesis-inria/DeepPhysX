from typing import Type, Tuple, Dict, Any, Union, List, Optional
from numpy import ndarray
from SSD.Core.Rendering.user_api import UserAPI

from DeepPhysX.database.database_handler import DatabaseHandler, Database


class AbstractController:

    compute_training_data: bool

    @property
    def environment_ids(self) -> Tuple[int, int]: raise NotImplementedError

    @property
    def database_handler(self) -> DatabaseHandler: raise NotImplementedError

    @property
    def visualization_factory(self) -> Optional[UserAPI]: raise NotImplementedError

    def create_environment(self) -> None: ...

    def save_parameters(self, **kwargs) -> None: ...

    def load_parameters(self) -> Dict[str, Any]: ...

    def define_database_fields(self, fields: Union[List[Tuple[str, Type]], Tuple[str, Type]]) -> None: ...

    def create_visualization(self, visualization_db: Union[Database, Tuple[str, str]], produce_data: bool = True) -> None: ...

    def connect_visualization(self) -> None: ...

    def set_data(self, **kwargs) -> None: ...

    def get_data(self) -> Dict[str, ndarray]: ...

    def get_prediction(self, **kwargs) -> Dict[str, ndarray]: ...

    def trigger_prediction(self) -> None: ...

    def trigger_send_data(self) -> List[int]: ...

    def trigger_update_data(self,  line_id: List[int]) -> None: ...

    def trigger_get_data(self, line_id: List[int]) -> None: ...

    def reset_data(self) -> None: ...
