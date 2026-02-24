"""
Document parser for the Analyst Agent.

Uses Docling to convert uploaded documents (PDF, DOCX, PPTX, XLSX, HTML, MD)
into hierarchical sections stored in the internal database.
"""
import os
import hashlib
import logging
from typing import List, Dict, Any, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.labels import DocItemLabel

from medds_agent.database import InternalDatabase

logger = logging.getLogger(__name__)

# File extensions that should be parsed for document search
PARSEABLE_EXTENSIONS = {
    '.pdf', '.docx', '.pptx', '.xlsx',
    '.html', '.htm', '.md', '.txt',
}


def _file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def is_parseable(file_path: str) -> bool:
    """Check if a file extension is supported for document parsing."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in PARSEABLE_EXTENSIONS


class DocumentParser:
    """
    Parses documents into hierarchical sections using Docling
    and stores them in the SQLite database.
    """

    def __init__(self, db: InternalDatabase):
        self.db = db
        self._converter: Optional[DocumentConverter] = None

    @property
    def converter(self) -> DocumentConverter:
        """Lazy-init the Docling converter (avoids import cost at startup)."""
        if self._converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False 
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                }
            )
        return self._converter

    def parse(self, session_id: str, file_path: str, file_name: str) -> bool:
        """
        Parse a document and store its sections in the database.

        Parameters
        ----------
        session_id : str
            The session this document belongs to.
        file_path : str
            Absolute path to the file on disk.
        file_name : str
            The logical file name (e.g., "data_dictionary.pdf").

        Returns
        -------
        bool
            True if parsing succeeded, False otherwise.
        """
        if not is_parseable(file_path):
            return False

        # Check if already parsed with same hash
        current_hash = _file_hash(file_path)
        existing = self.db.get_parsed_document(session_id, file_name)
        if existing and existing['file_hash'] == current_hash:
            return True  # Already up to date

        try:
            # Convert with Docling
            result = self.converter.convert(file_path)
            doc = result.document

            # Build hierarchical sections
            sections = self._extract_sections(doc)

            # Store in database
            document_id = self.db.upsert_parsed_document(
                session_id, file_name, current_hash
            )
            if sections:
                self.db.insert_sections(document_id, sections)

            logger.info(
                f"Parsed '{file_name}' for session {session_id}: "
                f"{len(sections)} sections extracted."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to parse '{file_name}': {e}")
            return False

    def _extract_sections(self, doc) -> List[Dict[str, Any]]:
        """
        Walk the Docling document tree and extract hierarchical sections.

        Strategy:
        - Iterate items; when a SectionHeader or Title is found, start a new section.
        - Non-heading items (Text, Table, ListItem, etc.) are appended to the
          current section's content.
        - The heading level from iterate_items() determines hierarchy.
        - If no headings are found, fall back to fixed-size chunking.
        """
        sections: List[Dict[str, Any]] = []
        # Stack tracks (level, section_index) for building hierarchy
        level_stack: List[tuple] = []
        current_section: Optional[Dict[str, Any]] = None
        order_counter = 0
        # Counters per level for generating section IDs like "1", "1.1", "1.2"
        level_counters: Dict[int, int] = {}

        heading_labels = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}

        for item, level in doc.iterate_items():
            item_label = item.label if hasattr(item, 'label') else None
            item_text = ""

            # Extract text content from the item
            if hasattr(item, 'text'):
                item_text = item.text
            elif hasattr(item, 'export_to_markdown'):
                try:
                    item_text = item.export_to_markdown()
                except Exception:
                    pass

            if not item_text or not item_text.strip():
                continue

            if item_label in heading_labels:
                # Flush current section
                if current_section is not None:
                    sections.append(current_section)

                # Reset counters for deeper levels
                for lvl in list(level_counters.keys()):
                    if lvl > level:
                        del level_counters[lvl]

                # Increment counter at this level
                level_counters[level] = level_counters.get(level, 0) + 1

                # Build section ID from counter chain: "1.2.3"
                section_id_parts = []
                for lvl in sorted(level_counters.keys()):
                    if lvl <= level:
                        section_id_parts.append(str(level_counters[lvl]))
                section_id = ".".join(section_id_parts)

                # Determine parent section ID
                # Pop stack until we find a level strictly less than current
                while level_stack and level_stack[-1][0] >= level:
                    level_stack.pop()

                parent_section_id = level_stack[-1][1] if level_stack else None

                current_section = {
                    "section_id": section_id,
                    "parent_section_id": parent_section_id,
                    "title": item_text.strip(),
                    "level": level,
                    "content": "",
                    "section_order": order_counter,
                }
                order_counter += 1

                level_stack.append((level, section_id))

            else:
                # Body content — append to current section
                if current_section is not None:
                    if current_section["content"]:
                        current_section["content"] += "\n" + item_text.strip()
                    else:
                        current_section["content"] = item_text.strip()
                else:
                    # Content before any heading — create a root section
                    current_section = {
                        "section_id": "0",
                        "parent_section_id": None,
                        "title": "(Introduction)",
                        "level": 0,
                        "content": item_text.strip(),
                        "section_order": order_counter,
                    }
                    order_counter += 1

        # Flush last section
        if current_section is not None:
            sections.append(current_section)

        # Fallback: if no headings were detected, chunk by fixed size
        if not sections or (len(sections) == 1 and sections[0]["section_id"] == "0"):
            sections = self._fallback_chunking(doc)

        return sections

    def _fallback_chunking(self, doc) -> List[Dict[str, Any]]:
        """
        Fallback for documents with no detectable headings.
        Collects all text and splits into fixed-size chunks (~800 tokens ≈ ~3200 chars)
        with ~100 token overlap (~400 chars).
        """
        all_text = []
        for item, level in doc.iterate_items():
            text = ""
            if hasattr(item, 'text'):
                text = item.text
            elif hasattr(item, 'export_to_markdown'):
                try:
                    text = item.export_to_markdown()
                except Exception:
                    pass
            if text and text.strip():
                all_text.append(text.strip())

        full_text = "\n".join(all_text)
        if not full_text:
            return []

        chunk_size = 3200  # ~800 tokens
        overlap = 400      # ~100 tokens
        sections = []
        start = 0
        order = 0

        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end]

            sections.append({
                "section_id": str(order + 1),
                "parent_section_id": None,
                "title": f"Chunk {order + 1}",
                "level": 1,
                "content": chunk,
                "section_order": order,
            })
            order += 1
            start = end - overlap
            if start < 0:
                start = 0

        return sections