class OcclusionClassifierError(Exception):
    pass


class CheckpointLoadError(OcclusionClassifierError):
    pass


class DatasetError(OcclusionClassifierError):
    pass
