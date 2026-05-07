import unittest
from argparse import Namespace

from tests.test_support import import_zimage_train_module


zimage_train = import_zimage_train_module()


class TestZImageSoarArgs(unittest.TestCase):
    def build_parser(self):
        parser = zimage_train.argparse.ArgumentParser()
        parser = zimage_train.zimage_finetune_setup_parser(parser)
        return parser

    def test_soar_defaults(self):
        args = self.build_parser().parse_args([])
        self.assertTrue(hasattr(args, "soar"))
        self.assertEqual(args.soar_lambda_aux, 1.0)
        self.assertEqual(args.soar_trajectory_length, 6)
        self.assertEqual(args.soar_num_sampling_steps, 40)

    def test_validate_soar_rejects_fused_backward(self):
        args = Namespace(
            soar=True,
            fused_backward_pass=True,
            soar_lambda_aux=1.0,
            soar_trajectory_length=6,
            soar_num_sampling_steps=40,
        )
        with self.assertRaises(ValueError):
            zimage_train.validate_soar_args(args)

    def test_validate_soar_rejects_non_positive_trajectory_length(self):
        args = Namespace(
            soar=True,
            fused_backward_pass=False,
            soar_lambda_aux=1.0,
            soar_trajectory_length=0,
            soar_num_sampling_steps=40,
        )
        with self.assertRaises(ValueError):
            zimage_train.validate_soar_args(args)


if __name__ == "__main__":
    unittest.main()
