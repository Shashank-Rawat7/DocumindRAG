from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class IndexingStatusResponse(BaseModel):
    """A document's indexing state, read from the database at one moment.

    Every field is a fact about stored rows, never a promise about retrieval:

    * ``indexing_status``: the last lifecycle state written (None = never queued).
      It does not say an attempt is running or that the latest attempt succeeded.
    * ``indexing_error``: the fixed message of the most recent failed attempt. It is
      cleared only by a successful publish, so it can remain on other states.
    * ``indexing_attempt``: how many claims have started.
    * ``indexed_at``: when the last successful publication committed (None if there
      is none).
    * ``index_current``: the published source version equals the current one.
    * ``published_chunks_present``: chunk rows exist for the document. It does not
      say they belong to the current publication or that retrieval will return them.
    * ``in_progress``: a claim exists whose lease has not expired. It does not say a
      worker is alive.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: int
    indexing_status: Optional[str]
    indexing_error: Optional[str]
    indexing_attempt: int
    indexed_at: Optional[datetime]
    index_current: bool
    published_chunks_present: bool
    in_progress: bool