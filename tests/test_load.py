from unittest import skip, TestCase
import scripts.load as load


class TestLoad(TestCase):
    """test class for loading features into S3
    """

    @classmethod
    def setUpClass(cls):
        """perform at test class initialization
        """
        cls.kwargs = dict(
            test=True
        )

    @classmethod
    def tearDownClass(cls):
        """perform when all tests are complete
        """
        pass

    def setUp(self):
        """perform before each unittest"""
        pass

    def tearDown(self):
        """perform after each unittest
        """
        pass

    def test_get_feature_names(self):
        features = load.get_feature_names(kwargs=self.kwargs)
        self.assertTrue(all([
            c in ['test_1', 'test_2'] for c in features
        ]))

    def test_validate_name(self):
        self.assertFalse(load.validate_name('test_1', kwargs=self.kwargs))
        self.assertTrue(load.validate_name('test_3', kwargs=self.kwargs))


    def test_upload_feature(self):
