from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class is for defining arguments during only testing."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--mode', type=str, default='test', help='test mode')
        parser.add_argument('--batch_size', type=int, default=1, help='the batch size for testing')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--threshold', type=float, default=0.2, help='threshold for metric of difference between real and fake images')
        parser.add_argument('--rotate', action='store_false', help='rotate image in tranforms')
        parser.add_argument('--brightness', default=0., type=float, help='change brightness of images in tranforms')
        return parser