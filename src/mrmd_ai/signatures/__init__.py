"""DSPy signature definitions for MRMD AI programs."""

from .finish import (
    FinishSentenceSignature,
    FinishParagraphSignature,
    FinishCodeLineSignature,
    FinishCodeSectionSignature,
)
from .fix import (
    FixGrammarSignature,
    FixTranscriptionSignature,
)
from .correct import (
    CorrectAndFinishLineSignature,
    CorrectAndFinishSectionSignature,
)
from .edit import (
    Edit,
    CommentInfo,
    EditAtCursorSignature,
    AddressCommentSignature,
    AddressAllCommentsSignature,
    AddressNearbyCommentSignature,
)

__all__ = [
    "FinishSentenceSignature",
    "FinishParagraphSignature",
    "FinishCodeLineSignature",
    "FinishCodeSectionSignature",
    "FixGrammarSignature",
    "FixTranscriptionSignature",
    "CorrectAndFinishLineSignature",
    "CorrectAndFinishSectionSignature",
    # Edit signatures
    "Edit",
    "CommentInfo",
    "EditAtCursorSignature",
    "AddressCommentSignature",
    "AddressAllCommentsSignature",
    "AddressNearbyCommentSignature",
]
