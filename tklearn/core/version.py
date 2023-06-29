from packaging import version

__all__ = [
    'Version',
]


class Version(version.Version):
    def __init__(
            self,
            version: str,
            description: str = None
    ) -> None:
        super().__init__(str(version))
        self.description = description

    def dict(self):
        return {
            'version': str(self),
            'description': self.description,
        }


class versions:
    latest = object()
    default = Version('0.0.1')
