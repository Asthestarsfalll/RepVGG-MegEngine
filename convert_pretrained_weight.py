import argparse
import megengine as mge
from verify import verifyRepVGG

mapper = {
    'rbr_1x1': 'pointwise',
    'rbr_dense': 'dense',
    'rbr_identity': 'identity',
}


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin-dir",
        default='./repvgg_a0.pkl',
        type=str,
        help='path to origin model dir'
    )
    parser.add_argument(
        "--save-dir",
        default='./converted_revgg_a0.pkl',
        type=str,
        help="path to save converted pkl file"
    )
    parser.add_argument(
        "--model-arch",
        default=None,
        type=str,
        help="model architecture(optional: RepVGGA0,RepVGGA1,RepVGGA2,RepVGGB0,RepVGGB1,RepVGGB1g2,RepVGGB1g4,RepVGGB2,RepVGGB2g2,RepVGGB2g4,RepVGGB3,RepVGGB3g2,RepVGGB3g4,RepVGGD2se)"
    )
    return parser


def convert(old_dict):
    new_dict = {}
    new_dict['linear.bias'] = old_dict.pop('head.fc.bias')
    new_dict['linear.weight'] = old_dict.pop('head.fc.weight')
    for key_origin, value_origin in old_dict.items():
        for key, value in mapper.items():
            if key in key_origin:
                new_dict[key_origin.replace(key, value)] = value_origin
    return new_dict



def main():
    parser = make_parser()
    args = parser.parse_args()
    state_dict = mge.load(args.origin_dir)
    new_dict = convert(state_dict)
    verifyRepVGG(args.model_name, new_dict)
    mge.save(new_dict, args.save_dir)


if __name__ == '__main__':
    main()
