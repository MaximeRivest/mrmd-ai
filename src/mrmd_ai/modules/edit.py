"""Edit modules for cursor-based editing and comment processing."""

import dspy
from typing import List, Optional
from ..signatures.edit import (
    Edit,
    CommentInfo,
    EditAtCursorSignature,
    AddressCommentSignature,
    AddressAllCommentsSignature,
    AddressNearbyCommentSignature,
)


class EditAtCursorPredict(dspy.Module):
    """Execute user instructions via precise find/replace edits."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(EditAtCursorSignature)

    def forward(
        self,
        text_before: str,
        text_after: str,
        selection: str,
        full_document: str,
        instruction: str,
    ):
        return self.predict(
            text_before=text_before,
            text_after=text_after,
            selection=selection,
            full_document=full_document,
            instruction=instruction,
        )


class AddressCommentPredict(dspy.Module):
    """Address a single comment embedded in the document."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(AddressCommentSignature)

    def forward(
        self,
        full_document: str,
        comment_text: str,
        comment_context_before: str,
        comment_context_after: str,
        comment_raw: str,
    ):
        return self.predict(
            full_document=full_document,
            comment_text=comment_text,
            comment_context_before=comment_context_before,
            comment_context_after=comment_context_after,
            comment_raw=comment_raw,
        )


class AddressAllCommentsPredict(dspy.Module):
    """Address all comments in a document."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(AddressAllCommentsSignature)

    def forward(
        self,
        full_document: str,
        comments: List[CommentInfo],
    ):
        return self.predict(
            full_document=full_document,
            comments=comments,
        )


class AddressNearbyCommentPredict(dspy.Module):
    """Address the comment nearest to the cursor."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(AddressNearbyCommentSignature)

    def forward(
        self,
        full_document: str,
        cursor_context_before: str,
        cursor_context_after: str,
        nearby_comment: CommentInfo,
        nearby_comment_raw: str,
    ):
        return self.predict(
            full_document=full_document,
            cursor_context_before=cursor_context_before,
            cursor_context_after=cursor_context_after,
            nearby_comment=nearby_comment,
            nearby_comment_raw=nearby_comment_raw,
        )
