from unittest import skip, TestCase

import pandas as pd

import dankypipe.pipe as pipe


class TestPipe(TestCase):
    """test class for loading features into S3
    """

    @classmethod
    def setUpClass(cls):
        """perform at test class initialization
        """
        cls.config = dict(
            job_name='test_job_1',
            features=['feature_3', 'feature_4'],
            model=dict(
                name='lightgbm',
                parameters=dict(
                    verbosity=1
                )
            ),
            task='tune',
            tuning=dict(
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
                ],
                metric='auc'
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
        features = pipe.get_feature_names(**self.kwargs)
        self.assertTrue(all([
            c in ['test_1', 'test_2'] for c in features
        ]))

    def test_validate_name(self):
        self.assertFalse(pipe.validate_name('test_1', **self.kwargs))
        self.assertTrue(pipe.validate_name('test-1', **self.kwargs))

    def test_upload_feature(self):
        self.assertRaises(
            ValueError,
            pipe.upload_feature,
            feature_name='test_1',
            datasets=('.', '.', '.'),
            **self.kwargs
        )
        self.assertRaises(
            FileNotFoundError,
            pipe.upload_feature,
            feature_name='test_3',
            paths=('test_3.csv', 'test_3.csv', 'test_3.csv'),
            **self.kwargs
        )
        paths = (
            'feature_3_train.csv',
            'feature_3_test.csv',
            'feature_3_validate.csv'
        )
        self.assertTrue(pipe.upload_feature(
            feature_name='feature_3',
            datasets=paths,
            overwrite=True,
            **self.kwargs
        ) is not None)
        self.assertRaises(
            ValueError,
            pipe.upload_feature,
            feature_name='feature_3',
            datasets=paths,
            **self.kwargs
        )

    def test_upload_feature_from_df(self):
        dfs = [
            pd.read_csv('feature_3_test.csv'),
            pd.read_csv('feature_3_train.csv'),
            pd.read_csv('feature_3_validate.csv')
        ]
        self.assertTrue(pipe.upload_feature(
            feature_name='feature_3',
            datasets=dfs,
            overwrite=True,
            **self.kwargs
        ) is not None)

    def test_download_feature(self):
        self.assertRaises(
            ValueError,
            pipe.download_feature,
            feature_name='test_3'
        )
        result = pipe.download_feature(
            'feature_3',
            **self.kwargs
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
            **self.kwargs
        )
        self.assertTrue(isinstance(result, dict))
        self.assertTrue(all([
            c in ['train', 'test', 'validate']
            for c in result.keys()
        ]))

    def test_prepare_job(self):
        ins = pipe.Ec2Job(config=self.config)

        self.assertRaises(
            ValueError,
            ins.prepare_job,
            **self.kwargs
        )

        result = ins.prepare_job(
            self.config,
            overwrite=True,
            **self.kwargs
        )
        self.assertTrue(
            'submit_time' in result.keys() and
            'status_code' in result.keys()
        )

    def test_get_jobs_listing(self):
        self.assertTrue('test_job_1' in pipe.get_jobs_listing(**self.kwargs))

    def test_download_config(self):
        self.assertRaises(
            ValueError,
            pipe.download_config,
            job_name='not_a_job'
        )

        result = pipe.download_config('test_job_1', **self.kwargs)
        self.assertTrue(result == self.config)

    def test_prep_init(self):
        result = pipe.prepare_init('test_job_1', **self.kwargs)
        self.assertTrue(all([
            c not in result for c in [
                '"!', '!#'
            ]
        ]))

    @skip
    def test_run_job(self):
        pipe.run_job('test_job_1', **self.kwargs)

    @skip
    def test_ec2_connect(self):
        with pipe.ec2sftp('ec2-174-129-137-102.compute-1.amazonaws.com') as svr:
            print()

    def test_upload_results(self):
        result_summary = "this is a test\nof stuff"
        predictions = pd.DataFrame([
            [1, 1],
            [2, 0],
            [3, 0],
            [4, 1]
        ], columns=['idx', 'pre'])

        pipe.upload_results(
            'test_job_1',
            result_summary,
            predictions,
            **self.kwargs
        )

    def test_get_results(self):
        results = pipe.get_results(
            'test_job_1', True, **self.kwargs
        )
        self.assertTrue('config' in results.keys())
        self.assertTrue('summary' in results.keys())
        self.assertTrue('predictions' in results.keys())
