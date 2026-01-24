"""Signature definitions for cursor-based editing and comment processing."""

import dspy
from pydantic import BaseModel, Field
from typing import List, Optional


class Edit(BaseModel):
    """A single find/replace edit operation.

    For insertions at cursor, use find="" and the text will be inserted
    at the cursor position.
    """
    find: str = Field(
        description="Exact text to find in document. Use empty string for insertion at cursor."
    )
    replace: str = Field(
        description="Text to replace the found text with, or text to insert if find is empty."
    )


class CommentInfo(BaseModel):
    """Information about a comment in the document."""
    text: str = Field(description="The comment text content")
    context_before: str = Field(description="Text immediately before the comment")
    context_after: str = Field(description="Text immediately after the comment")


class EditAtCursorSignature(dspy.Signature):
    """
    Execute a user instruction by generating precise find/replace edits.

    You are given the cursor context and a natural language instruction.
    Generate a list of edits that implement the instruction.

    CRITICAL RULES:
    1. Each edit has `find` (exact text to locate) and `replace` (replacement text)
    2. For INSERTIONS at cursor: use find="" - the replace text will be inserted at cursor
    3. For MODIFICATIONS: find must match the EXACT text in the document (character-for-character)
    4. find strings must be UNIQUE enough to match only the intended location
    5. Include surrounding context in find to ensure uniqueness (e.g., "def process_data(items)" not just "process_data")
    6. Edits are applied in order - earlier edits may shift positions of later ones

    Examples:
    - Instruction: "add a docstring" → find the function definition, replace with definition + docstring
    - Instruction: "rename x to count" → find=" x " (with spaces), replace=" count "
    - Instruction: "insert a comment here" → find="", replace="# comment\\n"
    - Instruction: "delete this line" → find="the line content\\n", replace=""

    When the user has selected text, that text is provided in `selection`.
    Prefer to operate on the selection when it's relevant to the instruction.
    """

    text_before: str = dspy.InputField(
        desc="Text immediately before the cursor (up to 500 characters for context)"
    )
    text_after: str = dspy.InputField(
        desc="Text immediately after the cursor (up to 500 characters for context)"
    )
    selection: str = dspy.InputField(
        desc="Currently selected text, or empty string if no selection"
    )
    full_document: str = dspy.InputField(
        desc="The complete document content for full context"
    )
    instruction: str = dspy.InputField(
        desc="User's natural language instruction for what to do"
    )

    edits: List[Edit] = dspy.OutputField(
        desc="List of find/replace edits to apply. Order matters - applied sequentially."
    )


class AddressCommentSignature(dspy.Signature):
    """
    Address a single comment/instruction embedded in the document.

    Comments are marked with <!--! comment text !--> syntax.
    The comment contains instructions or notes that should be addressed.
    Generate edits that fulfill the comment's request.

    After addressing, you may optionally remove the comment marker itself.

    Guidelines:
    - Read the comment carefully to understand what's requested
    - Look at the surrounding context to understand where changes should go
    - Generate precise edits that address the comment
    - If the comment asks for something that's already done, return empty edits
    - Consider removing the comment after addressing it (include that as an edit)
    """

    full_document: str = dspy.InputField(
        desc="The complete document content"
    )
    comment_text: str = dspy.InputField(
        desc="The text content of the comment (without the <!--! !--> markers)"
    )
    comment_context_before: str = dspy.InputField(
        desc="Text immediately before the comment marker"
    )
    comment_context_after: str = dspy.InputField(
        desc="Text immediately after the comment marker"
    )
    comment_raw: str = dspy.InputField(
        desc="The full raw comment including markers (e.g., '<!--! add error handling !-->')"
    )

    edits: List[Edit] = dspy.OutputField(
        desc="List of find/replace edits to address the comment"
    )


class AddressAllCommentsSignature(dspy.Signature):
    """
    Address ALL comments/instructions in a document.

    Scan the document for all <!--! ... !--> comment markers and generate
    edits that address each one.

    Guidelines:
    - Process comments in document order (top to bottom)
    - Each comment should be addressed appropriately
    - Comments that conflict should be resolved sensibly
    - After addressing, remove the comment markers
    - Return all edits as a single list (they'll be applied in order)
    """

    full_document: str = dspy.InputField(
        desc="The complete document content with embedded comments"
    )
    comments: List[CommentInfo] = dspy.InputField(
        desc="List of all comments found in the document with their context"
    )

    edits: List[Edit] = dspy.OutputField(
        desc="List of all find/replace edits to address all comments"
    )


class AddressNearbyCommentSignature(dspy.Signature):
    """
    Address the comment nearest to the cursor position.

    Find the comment that's closest to where the user's cursor is and
    generate edits to address that specific comment.

    Guidelines:
    - Focus only on the comment nearest to the cursor
    - Use the cursor context to identify which comment is relevant
    - Generate edits that address that comment
    - Optionally remove the comment marker after addressing
    """

    full_document: str = dspy.InputField(
        desc="The complete document content"
    )
    cursor_context_before: str = dspy.InputField(
        desc="Text before the cursor position"
    )
    cursor_context_after: str = dspy.InputField(
        desc="Text after the cursor position"
    )
    nearby_comment: CommentInfo = dspy.InputField(
        desc="The comment closest to the cursor"
    )
    nearby_comment_raw: str = dspy.InputField(
        desc="The full raw comment including markers"
    )

    edits: List[Edit] = dspy.OutputField(
        desc="List of find/replace edits to address the nearby comment"
    )
