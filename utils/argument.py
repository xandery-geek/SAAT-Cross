
def add_base_arguments(parser):
    # arguments for base config
    parser.add_argument('--device', dest='device', type=str, default='0', help='gpu device')
    return parser


def add_dataset_arguments(parser):
    
    # arguments for dataset
    parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
    parser.add_argument('--dataset', dest='dataset', default='FLICKR-25K',
                        choices=['FLICKR-25K', 'NUS-WIDE', 'MS-COCO'],
                        help='name of the dataset')
    return parser


def add_model_arguments(parser):
    # arguments for hashing model
    parser.add_argument('--hash_method', dest='hash_method', default='DCMH', 
                        choices=['DCMH', 'SSAH'],
                        help='names of deep hashing methods')
    parser.add_argument('--img_backbone', dest='img_backbone', default='ResNet18',
                        choices=['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet50'],
                        help='backbone network for image feature extrator')
    parser.add_argument('--txt_backbone', dest='txt_backbone', default='FC',
                        choices=['FC'],
                        help='backbone network for text feature extrator')
    parser.add_argument('--bit', dest='bit', type=int, default=32, help='length of the hashing code')
    return parser


def add_attack_arguments(parser):
    # arguments for attack
    parser.add_argument('--attack_method', dest='attack_method', default='mainstay', help='name of attack method')
    parser.add_argument('--generator', dest='generator', default='PGD', help='adversarial examples generator')
    parser.add_argument('--targeted', dest='targeted', action="store_true", default=False, help='targeted attack')
    parser.add_argument('--iteration', dest='iteration', type=int, default=100, help='iteration of adversarial attack')
    parser.add_argument('--img_eps', dest='img_eps', type=float, default=8/255., help='perturbation budget of image data')
    parser.add_argument('--txt_eps', dest='txt_eps', type=float, default=0.01, help='perturbation budget of text data')
    parser.add_argument('--retrieve', dest='retrieve', action="store_true", default=False, help='retrieve images')
    parser.add_argument('--sample', dest='sample', action="store_true", default=False, help='sample adversarial examples')
    return parser


def add_defense_arguments(parser):
    parser.add_argument('--defense_method', dest='defense_method', type=str, default='mainstay', choices=['mainstay'],
                        help='name of adversarial defense method')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--generator', dest='generator', default='PGD', help='adversarial examples generator')
    parser.add_argument('--iteration', dest='iteration', type=int, default=7, help='iteration of adversarial attack')
    parser.add_argument('--img_eps', dest='img_eps', type=float, default=8/255., help='perturbation budget of image data')
    parser.add_argument('--txt_eps', dest='txt_eps', type=float, default=0.01, help='perturbation budget of text data')
    parser.add_argument('--lam', dest='lam', type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument('--mu', dest='mu', type=float, default=1e-3, help='mu for quantization loss')
    return parser