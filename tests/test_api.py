"""Testing the API components of Frhodo"""


def test_launch(frhodo_app):
    """Make sure we can launch Frhodo"""
    app, window = frhodo_app
    assert window.isVisible()
