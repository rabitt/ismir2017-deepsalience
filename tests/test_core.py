import unittest
import os
import numpy as np

import imp

core = imp.load_source(
    'core', '../deepsalience/core.py')


class TestPatchSize(unittest.TestCase):

    def test_patch_size(self):
        expected = (360, 50)
        actual = core.patch_size()
        self.assertEqual(expected, actual)

class TestDataPaths(unittest.TestCase):

    def test_mf0_complete(self):
        expected = os.path.join(
            '/scratch/rmb456/multif0_ismir2017',
            'training_data_with_blur',
            'multif0_complete'
        )
        actual = core.data_path_multif0_complete()
        self.assertEqual(expected, actual)

    def test_mf0_incomplete(self):
        expected = os.path.join(
            '/scratch/rmb456/multif0_ismir2017',
            'training_data_with_blur',
            'multif0_incomplete'
        )
        actual = core.data_path_multif0_incomplete()
        self.assertEqual(expected, actual)


class TestTrackIdList(unittest.TestCase):

    def setUp(self):
        self.list = core.track_id_list()

    def test_length(self):
        expected = 320
        actual = len(self.list)
        self.assertEqual(expected, actual)

    def test_v1(self):
        self.assertTrue('MusicDelta_Rock' in self.list)

    def test_v2(self):
        self.assertTrue(
            'MidnightBlue_StarsAreScreaming' in self.list
        )

    def test_vextra(self):
        self.assertTrue(
            'Adele_SomeoneLikeYou' in self.list
        )


class TestKerasGenerator(unittest.TestCase):

    def test_generator_len2(self):
        data_list = [
            ('data/input1.npy', 'data/output1.npy'),
            ('data/input2.npy', 'data/output2.npy')
        ]
        input_patch_size = core.patch_size()
        gen = core.keras_generator(data_list, input_patch_size)
        i = 1
        for sample in gen:
            if i > 15:
                break
            self.assertEqual(2, len(sample))
            self.assertEqual((1, 360, 50, 6), sample[0].shape)
            self.assertEqual((1, 360, 50), sample[1].shape)
            i += 1

    def test_generator_len11(self):
        data_list = [
            ('data/input1.npy', 'data/output1.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy'),
            ('data/input2.npy', 'data/output2.npy')
        ]
        input_patch_size = core.patch_size()
        gen = core.keras_generator(data_list, input_patch_size)
        i = 1
        for sample in gen:
            if i > 15:
                break
            self.assertEqual(2, len(sample))
            self.assertEqual((1, 360, 50, 6), sample[0].shape)
            self.assertEqual((1, 360, 50), sample[1].shape)
            i += 1

class TestGrabPatchOutput(unittest.TestCase):

    def test_example(self):
        y_data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        actual = core.grab_patch_output(1, 0, 2, 2, y_data)
        expected = np.array([[
            [4, 5],
            [7, 8]
        ]])

        self.assertTrue(np.array_equal(expected, actual))

class TestGrabPatchInput(unittest.TestCase):

    def test_example(self):
        x_data = np.array([
            [
                [1, 2, 3],
                [4, 5, 6]
            ],
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        ])
        actual = core.grab_patch_input(0, 1, 2, 2, x_data)
        expected = np.array([[
            [[2, 8], [3, 9]],
            [[5, 11], [6, 12]]
        ]])

        self.assertTrue(np.array_equal(expected, actual))


class TestPatchGenerator(unittest.TestCase):

    def test_example(self):
        fpath_in = 'data/input1.npy'
        fpath_out = 'data/output1.npy'
        input_patch_size = core.patch_size()

        gen = core.patch_generator(fpath_in, fpath_out, input_patch_size)

        i = 1
        for sample in gen:
            if i > 15:
                break
            self.assertEqual(sorted(['X', 'Y']), sorted(sample.keys()))
            self.assertEqual((1, 360, 50, 6), sample['X'].shape)
            self.assertEqual((1, 360, 50), sample['Y'].shape)
            i += 1


class TestGetFilePaths(unittest.TestCase):

    def test_get_paths(self):
        mtrack_list = ['asdf', 'fdsa']
        data_path = 'data'
        actual = core.get_file_paths(mtrack_list, data_path)
        expected = [
            ('data/inputs/asdf_jklp_input.npy',
             'data/outputs/asdf_jklp_output.npy'),
            ('data/inputs/fdsa_jklp_input.npy',
             'data/outputs/fdsa_jklp_output.npy'),
        ]
        self.assertEqual(expected, actual)


class TestData(unittest.TestCase):

    def test_all(self):
        mtrack_list = core.track_id_list()
        data_path = 'data'
        input_patch_size = core.patch_size()
        dat = core.Data(mtrack_list, data_path, input_patch_size)
        self.assertTrue(len(dat.train_set) > 0)
        self.assertTrue(len(dat.validation_set) > 0)
        self.assertTrue(len(dat.test_set) > 0)
        self.assertTrue(len(dat.train_files) > 0)
        self.assertTrue(len(dat.validation_files) > 0)
        self.assertTrue(len(dat.test_files) > 0)



if __name__ == '__main__':
    unittest.main()
