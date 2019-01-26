from unittest import skip, TestCase

import pandas as pd

import scripts.pipe as pipe


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
        features = pipe.get_feature_names(kwargs=self.kwargs)
        self.assertTrue(all([
            c in ['test_1', 'test_2'] for c in features
        ]))

    def test_validate_name(self):
        self.assertFalse(pipe.validate_name('test_1', kwargs=self.kwargs))
        self.assertTrue(pipe.validate_name('test-1', kwargs=self.kwargs))

    def test_upload_feature(self):
        self.assertRaises(
            ValueError,
            pipe.upload_feature,
            feature_name='test_1',
            paths=('.', '.', '.'),
            kwargs=self.kwargs
        )
        self.assertRaises(
            FileNotFoundError,
            pipe.upload_feature,
            feature_name='test_3',
            paths=('test_3.csv', 'test_3.csv', 'test_3.csv'),
            kwargs=self.kwargs
        )
        paths = (
            'feature_3_train.csv',
            'feature_3_test.csv',
            'feature_3_validate.csv'
        )
        self.assertTrue(pipe.upload_feature(
            feature_name='feature_3',
            paths=paths,
            overwrite=True,
            kwargs=self.kwargs
        ) is None)
        self.assertRaises(
            ValueError,
            pipe.upload_feature,
            feature_name='feature_3',
            paths=paths,
            kwargs=self.kwargs
        )

    def test_download_feature(self):
        self.assertRaises(
            ValueError,
            pipe.download_feature,
            feature_name='test_3'
        )
        result = pipe.download_feature(
            'feature_3',
            kwargs=self.kwargs
        )
        self.assertTrue(isinstance(result, dict))
        self.assertTrue(all([
            c in ['train', 'test', 'validate']
            for c in result.keys()
        ]))
        self.assertTrue(all([
            isinstance(v, pd.DataFrame) and len(v.columns) == 2
            for k, v in result.items()
        ]))

    def test_build_feature_set(self):
        result = pipe.build_feature_set(
            ['feature_3', 'feature_4'],
            max_concurrent_conn=1,
            kwargs=self.kwargs
        )
        self.assertTrue(isinstance(result, dict))
        self.assertTrue(all([
            c in ['train', 'test', 'validate']
            for c in result.keys()
        ]))