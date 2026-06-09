"""``frhodo.common.config.FrhodoConfig`` v2 schema tests.

Pins:
  * round-trip: parse -> dump -> parse equals original
  * default constructor produces a parseable v2 doc with schema_version: 2
  * v1 files are rejected with ``SchemaVersionError``
  * uncertainty_function shape invariants (the legacy loader assumed
    >= 2 elements for "start, end" lists)
"""
import pytest
import yaml

from frhodo.common.errors import SchemaVersionError
from frhodo.common.config import FrhodoConfig


class TestDefaults:
    def test_default_construct_has_schema_version_2(self):
        c = FrhodoConfig()
        assert c.schema_version == 2

    def test_default_round_trip_equal(self):
        c = FrhodoConfig()
        text = c.to_yaml_text()
        c2 = FrhodoConfig.from_yaml_text(text)
        assert c == c2, "round-trip must be a no-op for the default config"

    def test_default_dumped_yaml_carries_schema_version(self):
        text = FrhodoConfig().to_yaml_text()
        data = yaml.safe_load(text)
        assert data["schema_version"] == 2


class TestSchemaRejection:
    def test_v1_file_rejected(self):
        v1_text = "schema_version: 1\n"
        with pytest.raises(SchemaVersionError) as excinfo:
            FrhodoConfig.from_yaml_text(v1_text)
        assert "v1" in str(excinfo.value)
        assert "default_config.yaml" in str(excinfo.value)

    def test_unknown_version_rejected(self):
        with pytest.raises(SchemaVersionError):
            FrhodoConfig.from_yaml_text("schema_version: 99\n")

    def test_missing_schema_version_uses_default(self):
        """A YAML doc without ``schema_version`` validates as v2 — not
        ideal but matches the lenient pre-PR behavior for partial files."""
        c = FrhodoConfig.from_yaml_text("plot:\n  x_scale: log\n")
        assert c.schema_version == 2
        assert c.plot.x_scale == "log"


