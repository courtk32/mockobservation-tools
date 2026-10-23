# Allow the user to check the version in a script
import importlib.metadata
try:
    __version__ = importlib.metadata.version("mockobservation_tools")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from . import galaxy_tools, general_tools, sersic_tools
