"""
Testing the functionality of the errors module
"""

import pytest
from pdstools.utils.errors import NotApplicableError


def test_not_applicable_error():
    """Test that the NotApplicableError class is defined correctly"""
    # Check that NotApplicableError is a subclass of ValueError
    assert issubclass(NotApplicableError, ValueError)
    
    # Check that NotApplicableError can be raised with a message
    with pytest.raises(NotApplicableError, match="Test error message"):
        raise NotApplicableError("Test error message")
    
    # Check that NotApplicableError can be caught as a ValueError
    try:
        raise NotApplicableError("Test error")
    except ValueError as e:
        assert isinstance(e, NotApplicableError)
        assert str(e) == "Test error"
