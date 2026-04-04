import ivis


def test_package_exports_version():
    assert ivis.__version__ == "0.1.0"


def test_package_exports_logger():
    assert ivis.logger.name == "IViS"
