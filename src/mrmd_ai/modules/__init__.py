"""DSPy module implementations for MRMD AI programs."""

from .finish import (
    FinishSentencePredict,
    FinishParagraphPredict,
    FinishCodeLinePredict,
    FinishCodeSectionPredict,
)
from .fix import (
    FixGrammarPredict,
    FixTranscriptionPredict,
)
from .correct import (
    CorrectAndFinishLinePredict,
    CorrectAndFinishSectionPredict,
)
from .code import (
    DocumentCodePredict,
    CompleteCodePredict,
    AddTypeHintsPredict,
    ImproveNamesPredict,
    ExplainCodePredict,
    RefactorCodePredict,
    FormatCodePredict,
    ProgramCodePredict,
)
from .text import (
    GetSynonymsPredict,
    GetPhraseSynonymsPredict,
    ReformatMarkdownPredict,
    IdentifyReplacementPredict,
)
from .document import (
    DocumentResponsePredict,
    DocumentSummaryPredict,
    DocumentAnalysisPredict,
)
from .notebook import (
    NotebookNamePredict,
)
from .edit import (
    EditAtCursorPredict,
    AddressCommentPredict,
    AddressAllCommentsPredict,
    AddressNearbyCommentPredict,
)

__all__ = [
    # Finish programs
    "FinishSentencePredict",
    "FinishParagraphPredict",
    "FinishCodeLinePredict",
    "FinishCodeSectionPredict",
    # Fix programs
    "FixGrammarPredict",
    "FixTranscriptionPredict",
    # Correct & Finish programs
    "CorrectAndFinishLinePredict",
    "CorrectAndFinishSectionPredict",
    # Code transformation programs
    "DocumentCodePredict",
    "CompleteCodePredict",
    "AddTypeHintsPredict",
    "ImproveNamesPredict",
    "ExplainCodePredict",
    "RefactorCodePredict",
    "FormatCodePredict",
    "ProgramCodePredict",
    # Text transformation programs
    "GetSynonymsPredict",
    "GetPhraseSynonymsPredict",
    "ReformatMarkdownPredict",
    "IdentifyReplacementPredict",
    # Document-level programs
    "DocumentResponsePredict",
    "DocumentSummaryPredict",
    "DocumentAnalysisPredict",
    # Notebook programs
    "NotebookNamePredict",
    # Edit programs (Ctrl-K and comments)
    "EditAtCursorPredict",
    "AddressCommentPredict",
    "AddressAllCommentsPredict",
    "AddressNearbyCommentPredict",
]
