from unittest import skip, TestCase

import pandas as pd

import dankypipe.runner as runner
import dankypipe.pipe as pipe


class TestJobRunner(TestCase):
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

    def test_fetch_data(self):
        config = runner.fetch_data(self.config['job_name'], **self.kwargs)
        self.assertTrue(config is not None)

    def test_runner(self):
        # config = pipe.download_config('a_small_demo')
        # runner.run_task(config)

        job = pipe.Ec2Job(config={
            'job_name': 'a_small_demo',
            'features': [
                'AvSigVersion', 'IsBeta'
            ],
            'model': {
                'name': 'lightgbm',
                'parameters': {
                    'verbosity': 1
                }
            },
            'task': 'tune',
            'tuning': {
                'search_type': 'grid',
                'parameters': [
                    {
                        'name': 'max_depth',
                        'values': [10, 20]
                    },
                    {
                        'name': 'learning_rate',
                        'values': [.1, .2, .3]
                    }
                ],
                'metric': 'auc'
            }
        }, overwrite=True)

        config = runner.fetch_data(job.config['job_name'])
        runner.run_task(config)
