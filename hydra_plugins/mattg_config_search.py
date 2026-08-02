from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class MattgSearchPathPlugin(SearchPathPlugin):
    """Automatically registers mattg's cfg directory with Hydra."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        from mattg import CONFIG_PATH
        search_path.append(
            provider="mattg",
            path=f"file://{CONFIG_PATH}",
        )
