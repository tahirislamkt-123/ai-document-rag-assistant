from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFProcessingError(ValueError):
    pass


def extract_pdf_pages(
    path,
    max_pages=200,
    max_text_characters=1000000,
):
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    if max_text_characters < 1:
        raise ValueError("max_text_characters must be positive")
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            raise PDFProcessingError("Encrypted PDFs are not supported")
        if not reader.pages:
            raise PDFProcessingError("PDF contains no pages")
        if len(reader.pages) > max_pages:
            raise PDFProcessingError(
                "PDF exceeds the {} page limit".format(max_pages)
            )

        pages = []
        character_count = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            character_count += len(text)
            if character_count > max_text_characters:
                raise PDFProcessingError(
                    "Extracted text exceeds the configured limit"
                )
            pages.append({"page": page_number, "text": text})
        return pages
    except PDFProcessingError:
        raise
    except (OSError, PdfReadError, ValueError, UnicodeError) as error:
        raise PDFProcessingError("PDF is unreadable or malformed") from error
