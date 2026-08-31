"""Unit tests for OCR field parsing (pure, no model or Flask needed)."""

from app.ai.ocr.extractor import parse_fields


def _lines():
    return [
        ("Bank Deposit Slip", 0.99),
        ("Name: Nimal Perera", 0.95),
        ("NIC 912345678V", 0.93),
        ("Account No 1234567890", 0.90),
        ("Amount Rs. 50,000.00", 0.88),
        ("Date 2025-08-16", 0.92),
    ]


class TestParseFields:
    def test_extracts_nic_uppercased(self):
        fields = parse_fields(_lines())["fields"]
        assert fields["nic_number"]["value"] == "912345678V"
        assert fields["nic_number"]["confidence"] == 0.93

    def test_extracts_labelled_name(self):
        fields = parse_fields(_lines())["fields"]
        assert fields["name"]["value"] == "Nimal Perera"

    def test_extracts_amount_and_date(self):
        fields = parse_fields(_lines())["fields"]
        assert fields["amount"]["value"] == "50,000.00"
        assert fields["date"]["value"] == "2025-08-16"

    def test_account_number_excludes_nic_and_dates(self):
        fields = parse_fields(_lines())["fields"]
        assert fields["account_no"]["value"] == "1234567890"

    def test_new_format_12_digit_nic(self):
        fields = parse_fields([("NIC 200012345678", 0.9)])["fields"]
        assert fields["nic_number"]["value"] == "200012345678"

    def test_mean_confidence_and_raw_text(self):
        result = parse_fields([("hello", 0.8), ("world", 0.6)])
        assert result["mean_confidence"] == 0.7
        assert result["raw_text"] == "hello\nworld"

    def test_empty_input(self):
        result = parse_fields([])
        assert result["fields"] == {}
        assert result["mean_confidence"] == 0.0
