import io
import tempfile
import unittest
from pathlib import Path

from backend.documents.chunking import chunk_pages
from backend.documents.pdf import PDFProcessingError, extract_pdf_pages
from tests.pdf_fixtures import make_text_pdf


class PDFAndChunkingTests(unittest.TestCase):
    def test_extracts_page_numbers_and_rejects_malformed_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.pdf"
            path.write_bytes(make_text_pdf("Annual leave is 25 days."))
            self.assertEqual(
                extract_pdf_pages(path),
                [{"page": 1, "text": "Annual leave is 25 days."}],
            )
            path.write_bytes(b"not a pdf")
            with self.assertRaises(PDFProcessingError):
                extract_pdf_pages(path)

    def test_page_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two-pages.pdf"
            path.write_bytes(make_text_pdf("One page."))

            with self.assertRaisesRegex(ValueError, "max_pages"):
                extract_pdf_pages(path, max_pages=0)

    def test_encrypted_pdf_is_rejected(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.encrypt("password")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encrypted.pdf"
            with path.open("wb") as output:
                writer.write(output)

            with self.assertRaisesRegex(PDFProcessingError, "Encrypted"):
                extract_pdf_pages(path)

    def test_chunks_are_stable_and_keep_page_provenance(self):
        page = {
            "page": 7,
            "text": "one two three four five six seven eight",
        }

        chunks = chunk_pages(
            [page],
            document_id="abc123",
            chunk_words=4,
            overlap_words=1,
        )

        self.assertEqual(
            [item["text"] for item in chunks],
            ["one two three four", "four five six seven", "seven eight"],
        )
        self.assertEqual([item["page"] for item in chunks], [7, 7, 7])
        self.assertEqual(
            [item["chunk_id"] for item in chunks],
            ["abc123:p7:c1", "abc123:p7:c2", "abc123:p7:c3"],
        )

    def test_invalid_chunk_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap_words"):
            chunk_pages(
                [{"page": 1, "text": "one"}],
                document_id="doc",
                chunk_words=4,
                overlap_words=4,
            )


if __name__ == "__main__":
    unittest.main()
