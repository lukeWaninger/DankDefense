import os
import sys
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
        cls.config = dict(
            job_name='test_job_1',
            features=['test_1', 'test_2'],
            validation='full',
            predict=dict(submit=False),
            parameter_tuning=dict(
                search_type='grid',
                parameters=[
                    dict(
                        name='max_depth',
                        values=[2]
                    ),
                    dict(
                        name='learning_rate',
                        values=[.1, .2, .3]
                    )
                ]
            )
        )
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
            c in ['train', 'test', 'validate'] and
            len(result[c].columns) == 3
            for c in result.keys()
        ]))

    def test_prepare_job(self):
        self.assertRaises(
            ValueError,
            pipe.prepare_job,
            config=self.config,
            overwrite=False,
            kwargs=self.kwargs
        )

        result = pipe.prepare_job(
            self.config,
            overwrite=True,
            kwargs=self.kwargs
        )
        self.assertTrue(
            'submit_time' in result.keys() and
            'status_code' in result.keys()
        )

    def test_get_jobs_listing(self):
        self.assertTrue('test_job_1' in pipe.get_jobs_listing(self.kwargs))

    def test_download_config(self):
        self.assertRaises(
            ValueError,
            pipe.download_config,
            job_name='not_a_job'
        )

        result = pipe.download_config('test_job_1', self.kwargs)
        self.assertTrue(result == self.config)

    def test_prep_init(self):
        result = pipe.prepare_init('test_job_1', kwargs=self.kwargs)
        self.assertTrue(all([
            c not in result for c in [
                '"!', '!#'
            ]
        ]))

    def test_run_job(self):
        pipe.run_job('test_job_1', kwargs=self.kwargs)

    def test_ec2_connect(self):
        with pipe.ec2connect('ec2-54-80-190-150.compute-1.amazonaws.com') as svr:
            print()
        print()