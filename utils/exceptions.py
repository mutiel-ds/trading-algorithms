class ProjectError(Exception):
    """
    Base class for all custom exceptions in this project.
    """
    pass

class DataNotFoundError(ProjectError):
    """
    Data not found or is empty.
    """
    pass