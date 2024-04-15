import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='UAGL', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--lev0', default=True, action='store_true', help='warp no')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=6)
    parser.add_argument('--val_freq', type=int, default=10000, help='validation frequency')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--eval_model', type=str, default=None)
    parser.add_argument('--resume', default=False, action='store_true', help='resume_training')
    parser.add_argument('--weight', action='store_true')
    parser.add_argument("--datasets_folder", type=str, default="datasets", help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, help="Relative path of the dataset")
    parser.add_argument("--prior_location_threshold", type=int, default=-1, help="The threshold of search region from prior knowledge for train and test. If -1, then no prior knowledge")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=50, help="_")
    parser.add_argument("--G_contrast", type=str, default="none", choices=["none", "manual", "autocontrast", "equalize"], help="G_contrast")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--permute", type=str, default="none", choices=["none", "img", "ue"])
    parser.add_argument("--database_size", type=int, default=512, choices=[512, 1024, 1536, 2048, 2560], help="database_size")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--two_stages", action="store_true", help="crop at level 2 but same scale")
    parser.add_argument("--fine_padding", type=int, default=0, help="expanding region of refinement")
    parser.add_argument("--corr_level", type=int, default=2, choices=[2, 4, 6], help="expanding region of refinement")
    parser.add_argument("--resize_width", type=int, default=256, choices=[256, 512], help="expanding region of refinement")
    parser.add_argument("--fnet_cat", action="store_true", help="fnet_cat")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--vis_all", action="store_true")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument('--augment_two_stages', type=float, default=0)
    parser.add_argument('--arch', type=str, default="IHN", choices=["IHN", "DHN", "LocalTrans"])
    parser.add_argument('--rotate_max', type=float, default=0)
    parser.add_argument('--resize_max', type=float, default=0)
    parser.add_argument('--permute_max', type=float, default=0)
    parser.add_argument('--first_stage_ue', action="store_true")
    parser.add_argument('--ue_shift', type=int, default=64)
    parser.add_argument('--ue_num_crops', type=int, default=5, choices=[5, 10])
    parser.add_argument('--ue_shift_crops_types', type=str, default="grid", choices=["grid", "random"])
    parser.add_argument('--ue_mask_prob', type=float, default=0.5)
    parser.add_argument('--ue_mask_patchsize', type=int, default=16)
    parser.add_argument('--ue_aug_method', type=str, default="shift", choices=["shift", "mask"])
    parser.add_argument('--ue_agg', type=str, choices=["mean", "zero", "maj_vote"], default="mean")
    parser.add_argument('--ue_rej_std', type=float, default=32.0)
    parser.add_argument('--ue_maj_vote_rej', type=float, default=32.0)
    parser.add_argument('--ue_mock', action="store_true")
    parser.add_argument("--ue_mock_loss_lambda", type=float, default=1.0, help="G_loss_lambda only for homo")
    args = parser.parse_args()
    args.save_dir = "local_he"
    args.augment_type = "center"
    return args