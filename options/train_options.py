from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class is for defining arguments during only training."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_epoch_freq', type=int, default=1, help='frequency of showing on screen')
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of save point')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting point of epoch if you have pretrained model, you can start from that point')
        parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
        parser.add_argument('--n_epochs_decay', type=int, default=500, help='the number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of Adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of Adam')
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--mode', type=str, default='train', help='train or pretrained mode')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout')
        parser.add_argument('--rotate', action='store_false', help='rotate image in tranforms')
        parser.add_argument('--brightness', default=0.1, type=float, help='change brightness of images in tranforms')
        return parser