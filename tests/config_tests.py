import unittest
from unittest.mock import MagicMock
from tklearn.config import Config, ConfigurableWrapper, config_cv


class TestConfigurableWrapper(unittest.TestCase):
    def setUp(self):
        self.builder = MagicMock()
        self.params = {"param1": "value1", "param2": "value2"}
        self.wrapper = ConfigurableWrapper(
            self.builder, name="builder", params=self.params
        )

    def test_set_params(self):
        new_params = {"param1": "new_value1", "param3": "value3"}
        new_wrapper = self.wrapper.set_params(**new_params)
        self.assertEqual(new_wrapper.params, Config(new_params))
