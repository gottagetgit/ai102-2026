"""
document_intelligence_prebuilt.py
==================================
Demonstrates using Azure Document Intelligence (formerly Form Recognizer)
prebuilt models to extract structured data from documents without training.

Prebuilt models covered:
  - prebuilt-invoice      : invoices (vendor, total, line items, tax, dates)
  - prebuilt-receipt      : receipts (merchant, items, subtotal, tax, total)
  - prebuilt-layout       : general layout extraction (paragraphs, tables, selection marks)
  - prebuilt-read         : OCR text extraction (all document types)
  - prebuilt-idDocument   : identity documents (passport, driver's license)
  - prebuilt-businessCard : business cards (name, email, phone, company)
  - prebuilt-contract     : contracts (parties, dates, payment terms)

Key SDK concepts:
  - analyze_document(): submits a document for analysis (URL or bytes)
  - DocumentAnalysisClient: the main analysis client
  - DocumentModelAdministrationClient: for listing, copying, composing models
  - AnalyzeResult: the root result object with pages, tables, fields, paragraphs

AI-102 Exam Skills Mapped:
  - Provision a Document Intelligence resource
  - Use prebuilt models to extract data from documents

Required environment variables (see .env.sample):
  AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT - https://<resource>.cognitiveservices.azure.com/
  AZURE_DOCUMENT_INTELLIGENCE_KEY      - API key

Package: azure-ai-documentintelligence>=1.0.0
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeResult,
    DocumentTable,
    DocumentPage,
)

load_dotenv()

ENDPOINT = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
KEY = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]

client = DocumentIntelligenceClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))

# Public sample documents hosted by Microsoft
SAMPLE_INVOICE_URL = (
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/"
    "master/curl/form-recognizer/sample-invoice.pdf"
)
SAMPLE_RECEIPT_URL = (
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/"
    "master/curl/form-recognizer/contoso-allinone.jpg"
)
SAMPLE_LAYOUT_URL = (
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/"
    "master/curl/form-recognizer/sample-layout.pdf"
)


# ---------------------------------------------------------------------------
# Helper: safely get field value
# ---------------------------------------------------------------------------
def get_field_value(fields: dict, field_name: str, default="N/A") -> str:
    """Extract a field value from a DocumentField dict, with a safe default."""
    field = fields.get(field_name)
    if field is None:
        return default
    value = field.get("valueString") or field.get("valueNumber") or \
            field.get("valueDate") or field.get("content")
    if value is None:
        return default
    return str(value)


# ---------------------------------------------------------------------------
# Demo 1: Invoice analysis
# ---------------------------------------------------------------------------
def demo_invoice_analysis(document_url: str = SAMPLE_INVOICE_URL):
    """
    Extract structured data from an invoice using the prebuilt-invoice model.

    Field categories returned:
      Vendor: VendorName, VendorAddress, VendorAddressRecipient
      Customer: CustomerName, CustomerAddress, BillingAddress
      Invoice: InvoiceId, InvoiceDate, DueDate, PurchaseOrder
      Financial: SubTotal, TotalTax, InvoiceTotal, AmountDue, PreviousUnpaidBalance
      Line items: Items[] with Description, Quantity, UnitPrice, Amount
    """
    print("\n" + "="*60)
    print(" Demo 1: Invoice Analysis (prebuilt-invoice)")
    print("="*60)
    print(f"Document: {document_url}")

    try:
        poller = client.begin_analyze_document(
            model_id="prebuilt-invoice",
            analyze_request=AnalyzeDocumentRequest(url_source=document_url),
            locale="en-US",
        )
        result: AnalyzeResult = poller.result()

        print(f"\nPages analyzed: {len(result.pages or [])}")
        print(f"Documents found: {len(result.documents or [])}")

        for doc_idx, document in enumerate(result.documents or []):
            print(f"\n--- Invoice {doc_idx + 1} ---")
            fields = document.fields or {}

            # Top-level invoice fields
            print(f"  Vendor Name    : {get_field_value(fields, 'VendorName')}")
            print(f"  Vendor Address : {get_field_value(fields, 'VendorAddress')}")
            print(f"  Invoice ID     : {get_field_value(fields, 'InvoiceId')}")
            print(f"  Invoice Date   : {get_field_value(fields, 'InvoiceDate')}")
            print(f"  Due Date       : {get_field_value(fields, 'DueDate')}")
            print(f"  Subtotal       : {get_field_value(fields, 'SubTotal')}")
            print(f"  Total Tax      : {get_field_value(fields, 'TotalTax')}")
            print(f"  Invoice Total  : {get_field_value(fields, 'InvoiceTotal')}")
            print(f"  Amount Due     : {get_field_value(fields, 'AmountDue')}")

            # Confidence scores — important for the exam
            if "InvoiceTotal" in fields:
                confidence = fields["InvoiceTotal"].get("confidence", "N/A")
                print(f"  InvoiceTotal confidence: {confidence}")

            # Line items (Items is a DocumentFieldType.Array of DocumentFieldType.Object)
            items_field = fields.get("Items")
            if items_field and items_field.get("valueArray"):
                print(f"\n  Line Items ({len(items_field['valueArray'])}):")
                for i, item in enumerate(items_field["valueArray"][:5], 1):
                    item_fields = item.get("valueObject", {})
                    desc = get_field_value(item_fields, "Description")
                    qty = get_field_value(item_fields, "Quantity")
                    unit_price = get_field_value(item_fields, "UnitPrice")
                    amount = get_field_value(item_fields, "Amount")
                    print(f"    [{i}] {desc} | Qty: {qty} | Unit: {unit_price} | Amount: {amount}")

    except HttpResponseError as e:
        print(f"  Error: {e.message} ({e.status_code})")


# ---------------------------------------------------------------------------
# Demo 2: Receipt analysis
# ---------------------------------------------------------------------------
def demo_receipt_analysis(document_url: str = SAMPLE_RECEIPT_URL):
    """
    Extract data from a retail receipt using prebuilt-receipt.

    Fields: MerchantName, MerchantAddress, MerchantPhoneNumber,
            TransactionDate, TransactionTime, Subtotal, Tax, Total, Tip
    Items[]: Name, Quantity, Price, TotalPrice
    """
    print("\n" + "="*60)
    print(" Demo 2: Receipt Analysis (prebuilt-receipt)")
    print("="*60)

    try:
        poller = client.begin_analyze_document(
            model_id="prebuilt-receipt",
            analyze_request=AnalyzeDocumentRequest(url_source=document_url),
        )
        result: AnalyzeResult = poller.result()

        for doc in result.documents or []:
            fields = doc.fields or {}
            print(f"\n  Merchant      : {get_field_value(fields, 'MerchantName')}")
            print(f"  Date          : {get_field_value(fields, 'TransactionDate')}")
            print(f"  Subtotal      : {get_field_value(fields, 'Subtotal')}")
            print(f"  Tax           : {get_field_value(fields, 'Tax')}")
            print(f"  Total         : {get_field_value(fields, 'Total')}")
            print(f"  Tip           : {get_field_value(fields, 'Tip')}")

    except HttpResponseError as e:
        print(f"  Error: {e.message}")


# ---------------------------------------------------------------------------
# Demo 3: Layout analysis — tables, paragraphs, selection marks
# ---------------------------------------------------------------------------
def demo_layout_analysis(document_url: str = SAMPLE_LAYOUT_URL):
    """
    prebuilt-layout extracts ALL structural content from a document:
      - Pages (dimensions, rotation)
      - Lines and words with bounding polygons
      - Tables (cell content, spans, row/column counts)
      - Selection marks (checkboxes, radio buttons) — state: selected/unselected
      - Paragraphs with semantic roles (title, section heading, footnote, etc.)

    This is the foundation that other prebuilt models build on.
    No domain-specific field extraction — use when you need raw structure.
    """
    print("\n" + "="*60)
    print(" Demo 3: Layout Analysis (prebuilt-layout)")
    print("="*60)

    try:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            analyze_request=AnalyzeDocumentRequest(url_source=document_url),
            # output_content_format="markdown"  # Optional: get output as Markdown
        )
        result: AnalyzeResult = poller.result()

        # Pages
        print(f"\nPages: {len(result.pages or [])}")
        for page in (result.pages or [])[:2]:
            print(
                f"  Page {page.page_number}: "
                f"{page.width}x{page.height} {page.unit} | "
                f"Lines: {len(page.lines or [])} | "
                f"Words: {len(page.words or [])}"
            )
            # Show first few lines of text
            for line in (page.lines or [])[:3]:
                print(f"    Line: '{line.content[:80]}'")

        # Tables
        print(f"\nTables: {len(result.tables or [])}")
        for tbl_idx, table in enumerate(result.tables or []):
            print(
                f"  Table {tbl_idx + 1}: "
                f"{table.row_count} rows x {table.column_count} cols"
            )
            # Print first few cells
            for cell in (table.cells or [])[:6]:
                print(
                    f"    [{cell.row_index},{cell.column_index}] "
                    f"(kind={cell.kind or 'content'}): '{cell.content[:60]}'"
                )

        # Selection marks (checkboxes)
        all_marks = [
            mark
            for page in (result.pages or [])
            for mark in (page.selection_marks or [])
        ]
        if all_marks:
            print(f"\nSelection Marks: {len(all_marks)}")
            for mark in all_marks[:5]:
                print(f"  State: {mark.state} | Confidence: {mark.confidence:.2f}")

        # Paragraphs with roles
        print(f"\nParagraphs: {len(result.paragraphs or [])}")
        for para in (result.paragraphs or [])[:5]:
            role = para.role or "body"
            print(f"  [{role}] '{para.content[:80]}'")

        # Key-value pairs (extracted without a custom model)
        if result.key_value_pairs:
            print(f"\nKey-Value Pairs: {len(result.key_value_pairs)}")
            for kv in result.key_value_pairs[:5]:
                key = kv.key.content if kv.key else "N/A"
                val = kv.value.content if kv.value else "N/A"
                confidence = kv.confidence
                print(f"  '{key}' → '{val}' (confidence: {confidence:.2f})")

    except HttpResponseError as e:
        print(f"  Error: {e.message}")


# ---------------------------------------------------------------------------
# Demo 4: Read model — pure OCR text extraction
# ---------------------------------------------------------------------------
def demo_read_model(document_url: str = SAMPLE_LAYOUT_URL):
    """
    prebuilt-read is the fastest model — pure text extraction with language detection.
    Use when you only need raw text, not structure or field extraction.

    Returns: pages, lines, words, languages detected per span.
    Does NOT return: tables, key-value pairs, or document fields.
    """
    print("\n" + "="*60)
    print(" Demo 4: Read Model (prebuilt-read) — OCR text extraction")
    print("="*60)

    try:
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            analyze_request=AnalyzeDocumentRequest(url_source=document_url),
        )
        result: AnalyzeResult = poller.result()

        print(f"\nContent length: {len(result.content or '')} characters")

        # Detected languages
        if result.languages:
            print("\nDetected languages:")
            for lang in result.languages[:3]:
                print(f"  {lang.locale} (confidence: {lang.confidence:.2f})")

        # Page-level text
        for page in (result.pages or [])[:1]:
            print(f"\nPage {page.page_number} text (first 500 chars):")
            full_page_text = " ".join(line.content for line in (page.lines or []))
            print(f"  {full_page_text[:500]}")

        # Document-level content (markdown-formatted if output_content_format="markdown")
        print("\nFull document content (first 500 chars):")
        print(f"  {(result.content or '')[:500]}")

    except HttpResponseError as e:
        print(f"  Error: {e.message}")


# ---------------------------------------------------------------------------
# Demo 5: List available prebuilt models
# ---------------------------------------------------------------------------
def demo_list_models():
    """
    DocumentModelAdministrationClient lets you list, get, delete, and
    copy document models. Use this to explore available prebuilt models.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceAdministrationClient

    admin_client = DocumentIntelligenceAdministrationClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )

    print("\n" + "="*60)
    print(" Demo 5: List Available Document Intelligence Models")
    print("="*60)

    try:
        models = admin_client.list_models()
        print(f"\nAvailable models:")
        for model in models:
            created = model.created_on.strftime("%Y-%m-%d") if model.created_on else "N/A"
            print(f"  {model.model_id:<35} created: {created}")

        # Resource info (quota and limits)
        resource_info = admin_client.get_resource_info()
        print(f"\nResource Info:")
        print(f"  Custom model limit : {resource_info.custom_document_models.limit}")
        print(f"  Custom models used : {resource_info.custom_document_models.count}")

    except HttpResponseError as e:
        print(f"  Error: {e.message}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Azure Document Intelligence — Prebuilt Model Demonstrations")
    print(f"Endpoint: {ENDPOINT}")

    try:
        demo_list_models()
        demo_invoice_analysis()
        demo_receipt_analysis()
        demo_layout_analysis()
        demo_read_model()
        print("\nAll prebuilt model demos complete!")

    except KeyError as e:
        print(f"Missing environment variable: {e}")
        raise
    except HttpResponseError as e:
        print(f"API error [{e.status_code}]: {e.message}")
        raise


if __name__ == "__main__":
    main()
