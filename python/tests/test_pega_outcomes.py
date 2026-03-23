"""Tests for the pega_outcomes shared utility."""

from pdstools.utils.pega_outcomes import resolve_outcome_labels


class TestResolveOutcomeLabels:
    def test_web_inbound_uses_clicked_as_accepts(self):
        result = resolve_outcome_labels({"Web/Inbound": ["Impression", "Clicked", "Rejected"]})
        assert result["Web/Inbound"]["Impressions"] == ["Impression"]
        assert result["Web/Inbound"]["Accepts"] == ["Clicked"]

    def test_call_center_uses_accepted_not_clicked(self):
        result = resolve_outcome_labels({"Call Center/Inbound": ["Impression", "Accepted", "Clicked", "Rejected"]})
        assert result["Call Center/Inbound"]["Impressions"] == ["Impression"]
        assert "Accepted" in result["Call Center/Inbound"]["Accepts"]
        assert "Clicked" not in result["Call Center/Inbound"]["Accepts"]

    def test_retail_uses_accepted(self):
        result = resolve_outcome_labels({"Retail/Inbound": ["Impression", "Accepted"]})
        assert result["Retail/Inbound"]["Accepts"] == ["Accepted"]

    def test_direct_mail_uses_pending_as_impressions(self):
        result = resolve_outcome_labels({"Direct Mail/Outbound": ["Pending", "Clicked"]})
        assert result["Direct Mail/Outbound"]["Impressions"] == ["Pending"]
        assert result["Direct Mail/Outbound"]["Accepts"] == ["Clicked"]

    def test_email_uses_digital_defaults(self):
        result = resolve_outcome_labels({"Email/Outbound": ["Impression", "Clicked", "Pending"]})
        assert result["Email/Outbound"]["Impressions"] == ["Impression"]
        assert result["Email/Outbound"]["Accepts"] == ["Clicked"]

    def test_filters_to_available_outcomes_only(self):
        # Clicked not in data for Web — must not appear in result
        result = resolve_outcome_labels({"Web/Inbound": ["Impression", "Rejected"]})
        assert result["Web/Inbound"]["Accepts"] == []

    def test_old_spelling_variants_included(self):
        result = resolve_outcome_labels({"Call Center/Inbound": ["Impression", "Accept", "Accepted"]})
        assert "Accept" in result["Call Center/Inbound"]["Accepts"]
        assert "Accepted" in result["Call Center/Inbound"]["Accepts"]

    def test_multiple_channels_resolved_independently(self):
        result = resolve_outcome_labels(
            {
                "Web/Inbound": ["Impression", "Clicked"],
                "Call Center/Inbound": ["Impression", "Accepted"],
            }
        )
        assert result["Web/Inbound"]["Accepts"] == ["Clicked"]
        assert result["Call Center/Inbound"]["Accepts"] == ["Accepted"]

    def test_unknown_channel_falls_back_to_digital_defaults(self):
        result = resolve_outcome_labels({"Custom/Inbound": ["Impression", "Clicked"]})
        assert result["Custom/Inbound"]["Impressions"] == ["Impression"]
        assert result["Custom/Inbound"]["Accepts"] == ["Clicked"]

    def test_empty_input_returns_empty_dict(self):
        assert resolve_outcome_labels({}) == {}
