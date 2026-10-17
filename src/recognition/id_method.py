"""How a track got its identity, and how strong that evidence is.

The single list of identification methods. Strength decides what may overwrite
what (`TrackIdentityState.confirm_identity`): a weaker method never replaces a
stronger one, while face evidence can still correct an appearance-based label.
"""

from enum import Enum


class IdMethod(str, Enum):
    NONE = "none"
    HANDOVER = "handover"          # copied from a track in a shared zone on another camera
    REID_GALLERY = "reid_gallery"  # matched the shared Re-ID (appearance) gallery
    FACE = "face"                  # confirmed by face recognition

    def __str__(self) -> str:  # f-strings, logs, JSON: the plain value
        return self.value

    @property
    def strength(self) -> int:
        return _STRENGTH[self]

    @property
    def is_appearance(self) -> bool:
        """Identity rests on appearance/location, not on a face (face may still upgrade it)."""
        return self in (IdMethod.HANDOVER, IdMethod.REID_GALLERY)


_STRENGTH = {IdMethod.NONE: 0, IdMethod.HANDOVER: 1, IdMethod.REID_GALLERY: 2, IdMethod.FACE: 3}
