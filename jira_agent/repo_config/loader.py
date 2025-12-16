"""Repository configuration loader."""

from pathlib import Path

import yaml

from .schema import RepoConfig


class ConfigLoader:
    """Load and manage repository configurations."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize config loader.

        Args:
            config_dir: Directory containing repo config YAML files.
                       Defaults to ./configs relative to package root.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs"
        self.config_dir = config_dir
        self._cache: dict[str, RepoConfig] = {}

    def load_from_file(self, config_path: Path | str) -> RepoConfig:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Parsed RepoConfig object.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValidationError: If config is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return RepoConfig.model_validate(data)

    def load_for_repo(self, repo_full_name: str) -> RepoConfig:
        """Load configuration for a repository.

        Looks for a config file matching the repo name in the config directory.
        For example, 'acme/data' looks for 'acme-data.yaml'.

        Args:
            repo_full_name: Full repository name (owner/name).

        Returns:
            Parsed RepoConfig object.

        Raises:
            FileNotFoundError: If no config found for repo.
        """
        if repo_full_name in self._cache:
            return self._cache[repo_full_name]

        # Convert owner/name to filename format
        config_filename = repo_full_name.replace("/", "-") + ".yaml"
        config_path = self.config_dir / config_filename

        if not config_path.exists():
            # Try .yml extension
            config_path = self.config_dir / (repo_full_name.replace("/", "-") + ".yml")

        if not config_path.exists():
            raise FileNotFoundError(
                f"No config found for repo '{repo_full_name}'. "
                f"Expected at: {self.config_dir / config_filename}"
            )

        config = self.load_from_file(config_path)
        self._cache[repo_full_name] = config
        return config

    def list_configs(self) -> list[str]:
        """List all available repository configurations.

        Returns:
            List of repository names that have configs.
        """
        configs = []
        if not self.config_dir.exists():
            return configs

        for config_file in self.config_dir.glob("*.yaml"):
            # Skip example config
            if config_file.stem == "example":
                continue
            # Convert filename back to repo name
            repo_name = config_file.stem.replace("-", "/", 1)
            configs.append(repo_name)

        for config_file in self.config_dir.glob("*.yml"):
            if config_file.stem == "example":
                continue
            repo_name = config_file.stem.replace("-", "/", 1)
            if repo_name not in configs:
                configs.append(repo_name)

        return sorted(configs)

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
