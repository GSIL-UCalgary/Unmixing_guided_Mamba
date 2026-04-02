import os
import time
import torch
import random
import argparse
import numpy as np
import wandb
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchvision import transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, SADLoss
from utils.evaluation import Evaluator
from utils.HSICommonUtils import ImageStretching
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
#from model.new_ import DSNet, show_images, show_endmember
from model.Unmixing_guided import UTKMamba, show_images, show_endmember, TemporalAbundanceEMA, temporal_abundance_loss_with_mask, sigmoid_rampup, show_W_variants

torch.autograd.set_detect_anomaly(True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_index", type=int, default=13)
    parser.add_argument("--data_set_path", type=str, default="./data")
    parser.add_argument("--work_dir", type=str, default="./")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--query_lr", type=float, default=5e-4)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--train_samples", type=int, default=30)
    parser.add_argument("--val_samples", type=int, default=10)
    parser.add_argument("--exp_name", type=str, default="RUNS")

    parser.add_argument("--wandb_project", type=str, default="Unmixing_guided_Mamba_HSIC_IN")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--endm_num", type=int, default=23)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--val_interval", type=int, default=50)

    return parser.parse_args()


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(
        gt_vis,
        pred_vis,
        save_single_predict_path,
        save_single_gt_path,
        only_vis_label=only_vis_label,
    )
    visualize_predict(
        gt_vis,
        pred_vis,
        save_single_predict_path.replace(".png", "_mask.png"),
        save_single_gt_path,
        only_vis_label=True,
    )


def freeze_endmember(net):
    for p in net.query_embed.parameters():
        p.requires_grad = False


def unfreeze_endmember(net):
    for p in net.query_embed.parameters():
        p.requires_grad = True

import matplotlib.pyplot as plt


def plot_initialized_queries(init_weight, save_path="init_queries.png"):
    """
    init_weight: [num_queries, band], torch tensor or numpy array
    """
    if isinstance(init_weight, torch.Tensor):
        data = init_weight.detach().cpu().numpy()
    else:
        data = init_weight

    fig = plt.figure(figsize=(6, 4))
    for i in range(data.shape[0]):
        plt.plot(data[i], linewidth=0.8)

    plt.xlabel("Bands", fontsize=14)
    plt.ylabel("Reflectance", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    args = get_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed_list = [0]
    num_list = [args.train_samples, args.val_samples]
    ratio_list = [0.01, 0.01]
    flag_list = [1, 0]

    net_name = "MambaHSI"
    data_set_name_list = ["IN", "UP", "HanChuan", "HongHu", "Houston", "LN", "SA", "blueberry", "QUH", "YYC", "Berlin", 'UAV', 'HU', 'Tea']
    data_set_name = data_set_name_list[args.dataset_index]
    split_image = data_set_name in ["HanChuan", "Houston"]

    save_folder = os.path.join(args.work_dir, args.exp_name, net_name, data_set_name)
    os.makedirs(save_folder, exist_ok=True)

    save_log_path = os.path.join(save_folder, f"train_tr{num_list[0]}_val{num_list[1]}.log")
    logger = setup_logger(name=data_set_name, logfile=save_log_path)
    logger.info(save_folder)

    data, gt = data_load_operate.load_data(data_set_name, args.data_set_path)
    height, width, channels = data.shape
    gt_reshape = gt.reshape(-1)

    img = ImageStretching(data)
    vca_end = data_load_operate.hyperVca(
        img.transpose(2, 0, 1).reshape(channels, height * width),
        args.endm_num
    )[0]

    class_count = int(np.max(gt))
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    SAD_func = SADLoss()

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []

    paras_dict = {
        "net_name": net_name,
        "dataset_index": args.dataset_index,
        "dataset_name": data_set_name,
        "num_list": num_list,
        "lr": args.lr,
        "query_lr": args.query_lr,
        "seed_list": seed_list,
        "endm_num": args.endm_num,
        "hidden_dim": args.hidden_dim,
    }
    logger.info(paras_dict)

    x = torch.from_numpy(img).unsqueeze(0).float().permute(0, 3, 1, 2)

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)

        single_experiment_name = f"run{exp_idx}_seed{curr_seed}"
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        save_vis_folder = os.path.join(save_single_experiment_folder, "vis")
        os.makedirs(save_single_experiment_folder, exist_ok=True)
        os.makedirs(save_vis_folder, exist_ok=True)

        save_weight_path = os.path.join(save_single_experiment_folder, f"best_tr{num_list[0]}_val{num_list[1]}.pth")
        results_save_path = os.path.join(save_single_experiment_folder, f"result_tr{num_list[0]}_val{num_list[1]}.txt")
        predict_save_path = os.path.join(save_single_experiment_folder, f"pred_vis_tr{num_list[0]}_val{num_list[1]}.png")
        gt_save_path = os.path.join(save_single_experiment_folder, f"gt_vis_tr{num_list[0]}_val{num_list[1]}.png")

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=f"{data_set_name}_{single_experiment_name}",
            config={
                "dataset_name": data_set_name,
                "dataset_index": args.dataset_index,
                "train_samples": args.train_samples,
                "val_samples": args.val_samples,
                "lr": args.lr,
                "query_lr": args.query_lr,
                "max_epoch": args.max_epoch,
                "endm_num": args.endm_num,
                "hidden_dim": args.hidden_dim,
                "seed": curr_seed,
                "split_image": split_image,
            },
            reinit=True,
        )

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(
            ratio_list,
            num_list,
            gt_reshape,
            class_count,
            flag_list[1]
        )

        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(
            data, height, width, gt_reshape, index
        )

        train_label = train_label.to(device)
        val_label = val_label.to(device)
        test_label = test_label.to(device)
        net = UTKMamba(channels, args.hidden_dim, args.endm_num, class_count)
        net.to(device)
        warmup_epoch = 100
        rampup_length = args.max_epoch // 10
        alpha = 0.9
        lambda_temp = 0.01
        conf_thresh = 0.6
        ema_helper = TemporalAbundanceEMA(alpha=alpha)

        vca_end_tensor = torch.tensor(vca_end, dtype=torch.float32).to(device)
        init_weight = net.init_query_embed_from_vca(vca_end_tensor.T, return_init=True)

        show_endmember(init_weight.cpu().numpy().reshape(args.endm_num, net.num_queries_times, channels), save_path=data_set_name + "_init_queries.png")
        #freeze_endmember(net=net)

        logger.info(net)
        wandb.watch(net, log="gradients", log_freq=50)

        query_params = list(net.query_embed.parameters())
        # unmix_params = list(net.unmix_decoder.parameters())
        weight_params = [net.weights]

        excluded_params = set(map(id, query_params + weight_params))
        base_params = [p for p in net.parameters() if id(p) not in excluded_params]

        optimizer = torch.optim.Adam(
            [
                {
                    "params": base_params,
                    "lr": args.lr,
                    "weight_decay": 0.0
                },
                {
                    "params": net.query_embed.parameters(),
                    "lr": args.query_lr,
                    "weight_decay": 0.0
                },
                {
                    "params": [net.weights],
                    "lr": args.query_lr,
                    "weight_decay": 0.0
                }
            ]
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, args.max_epoch // 10),
            gamma=0.9
        )

        freeze_epochs = 100
        step_size = max(1, args.max_epoch // 10)
        gamma = 0.9

        def lambda_base(epoch):
            # same behavior as StepLR for base params
            return gamma ** (epoch // step_size)

        def lambda_query(epoch):
            # keep query lr unchanged before freeze_epochs
            if epoch < freeze_epochs:
                return 1.0
            # start decay only after freeze_epochs
            return gamma ** ((epoch - freeze_epochs) // step_size)

        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=[lambda_base, lambda_query, lambda_query]
        # )

        logger.info(optimizer)

        best_val_acc = 0.0
        best_epoch = -1
        evaluator = Evaluator(num_class=class_count)

        tic_train = time.perf_counter()

        for epoch in range(args.max_epoch):
            net.train()
            y_train = train_label.unsqueeze(0)

            if split_image:
                x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                # recon1, output1, abu1, endm1 = net(x_part1)

                recon_linear, output_cls, pred_aban, endm_get, _, _ = net(x_part1.to(device))
                loss_cls1 = head_loss(loss_func, output_cls, y_part1.long())
                # recon_loss = torch.nn.functional.mse_loss(recon_linear, x_part1)
                loss_recon1 = SAD_func(recon_linear, x_part1.to(device))
                loss_abu1 = 0.001 * torch.norm(pred_aban, p=0.5, dim=1).mean()
                loss1 = loss_cls1 + 0.01 * loss_recon1 + loss_abu1


                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()

                recon_linear, output_cls, pred_aban, endm_get, _, _ = net(x_part2.to(device))
                loss_cls2 = head_loss(loss_func, output_cls, y_part2.long())
                # recon_loss = torch.nn.functional.mse_loss(recon_linear, x_part2)
                loss_recon2 = SAD_func(recon_linear, x_part2.to(device))
                loss_abu2 = 0.001 * torch.norm(pred_aban, p=0.5, dim=1).mean()
                loss2 = loss_cls2 + 0.01 * loss_recon2 + loss_abu2

                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()

                total_loss = loss1 + loss2
                cls_loss = loss_cls1 + loss_cls2
                recon_loss = loss_recon1 + loss_recon2
                abu_loss = loss_abu1 + loss_abu2

            else:
                optimizer.zero_grad()
                # abu_teacher = None
                token_beta = 0.0

                # if epoch >= freeze_epochs:
                abu_teacher = ema_helper.get_target()

                if abu_teacher is not None:
                    token_beta = sigmoid_rampup(epoch, rampup_length)
                    token_beta = 0.3 * token_beta
                        # 最大只混到 0.5，别让 teacher 完全主导
                    
                recon_linear, output_cls, pred_aban, endm_get, _, _ = net(x.to(device),abu_teacher=abu_teacher,token_beta=token_beta)
                #recon_linear, output_cls, pred_aban, endm_get, _, _ = net(x)
                cls_loss = head_loss(loss_func, output_cls, y_train.long())
                recon_loss = torch.nn.functional.mse_loss(recon_linear, x.to(device))
                sad_loss = SAD_func(recon_linear, x.to(device))
                abu_loss = 0.001 * torch.norm(pred_aban, p=0.5, dim=1).mean()
                total_loss = cls_loss + 0.01 * sad_loss + abu_loss
                            # temporal abundance loss
                # if epoch >= freeze_epochs and abu_teacher is not None:
                #     temp_weight = lambda_temp * sigmoid_rampup(epoch - freeze_epochs, rampup_length)
                #     loss_temp = temporal_abundance_loss_with_mask(
                #         abu=pred_aban,
                #         abu_target=abu_teacher,
                #         conf_thresh=conf_thresh,
                #     )
                #     total_loss = total_loss + temp_weight * loss_temp
                # else:
                #     loss_temp = torch.tensor(0.0, device=device)
        
                total_loss.backward()
                optimizer.step()
                # if epoch + 1 == freeze_epochs:
                #     unfreeze_endmember(net=net)
                ema_helper.update(pred_aban)

            scheduler.step()

            logger.info(
                f"Iter:{epoch} | total:{total_loss.item():.6f} | "
                f"cls:{cls_loss.item():.6f} | recon:{recon_loss.item():.6f} | "
                f"abu:{abu_loss.item():.6f}"
            )

            wandb.log({
                "epoch": epoch + 1,
                "train/total_loss": total_loss.item(),
                "train/cls_loss": cls_loss.item(),
                "train/recon_loss": recon_loss.item(),
                "train/abu_loss": abu_loss.item(),
                "train/lr_main": optimizer.param_groups[0]["lr"],
                "train/lr_query": optimizer.param_groups[1]["lr"],
            })

            if (epoch + 1) % args.val_interval == 0:
                net.eval()
                with torch.no_grad():
                    evaluator.reset()
                    y_val = val_label.unsqueeze(0)
                    abu_teacher = ema_helper.get_target()

                    token_beta = sigmoid_rampup(epoch, rampup_length)
                    token_beta = 0.3 * token_beta

                    recon_val, output_val, abu_val, endm_val, end_var_val, W_logits_ = net(x.to(device), abu_teacher=abu_teacher,token_beta=token_beta)
                    predict = torch.argmax(output_val, dim=1).cpu().numpy()

                    Y_val_np = val_label.cpu().numpy()
                    Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)

                    evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), predict)

                    OA = evaluator.Pixel_Accuracy()
                    mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                    mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                    Kappa = evaluator.Kappa()

                    logger.info(
                        f"Evaluate {epoch} | OA:{OA:.6f} | AA:{mAcc:.6f} | "
                        f"Kappa:{Kappa:.6f} | mIOU:{mIOU:.6f}"
                    )

                    if OA >= best_val_acc:
                        best_val_acc = OA
                        best_epoch = epoch + 1
                        torch.save(net.state_dict(), save_weight_path)
                        logger.info(f"Saved best model at epoch {best_epoch}")

                        wandb.summary["best_val_OA"] = best_val_acc
                        wandb.summary["best_epoch"] = best_epoch

                    save_single_predict_path = os.path.join(
                        save_vis_folder, f"predict_{epoch + 1}.png"
                    )
                    save_single_gt_path = os.path.join(save_vis_folder, "gt.png")
                    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)
                    abu_vis_path = os.path.join(save_single_experiment_folder, f"abu_val_epoch_{epoch+1}.png")
                    endm_vis_path = os.path.join(save_single_experiment_folder, f"endmember_val_epoch_{epoch+1}.png")
                    weights_val_path = os.path.join(save_single_experiment_folder, f"weights_val_epoch_{epoch+1}.png")

                    show_images(abu_val.detach(), save_path=abu_vis_path)
                    show_endmember(end_var_val.detach().cpu().numpy(), save_path=endm_vis_path)
                    show_W_variants(W_logits_, b=0, p=0, save_path=weights_val_path)
                    wandb.log({
                        "val/OA": OA,
                        "val/AA": mAcc,
                        "val/Kappa": Kappa,
                        "val/mIOU": mIOU,
                        "val/prediction_map": wandb.Image(save_single_predict_path, caption=f"Epoch {epoch+1}"),
                        "val/abundance_map": wandb.Image(abu_vis_path, caption=f"Epoch {epoch+1}"),
                        "val/endmember_plot": wandb.Image(endm_vis_path, caption=f"Epoch {epoch+1}"),
                    }, step=epoch + 1)

        train_time = time.perf_counter() - tic_train
        Train_Time_ALL.append(train_time)

        logger.info("\n==================== Starting evaluation for testing set ====================\n")

        #best_net = DSNet(channels, args.hidden_dim, args.endm_num, class_count, "conv2d_unmix")
        #best_net.to(device)
        net.load_state_dict(torch.load(save_weight_path))
        net.eval()

        test_evaluator = Evaluator(num_class=class_count)

        tic_test = time.perf_counter()
        with torch.no_grad():
            test_evaluator.reset()
            abu_teacher = ema_helper.get_target()
            token_beta = sigmoid_rampup(epoch, rampup_length)
            token_beta = 0.3 * token_beta

            recon_test, output_test, abu_test, endm_test, end_var_test, W_logits_ = net(x.to(device), abu_teacher=abu_teacher,token_beta=token_beta)

            predict_test = torch.argmax(output_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)

            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)

            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = test_evaluator.Kappa()

            logger.info(
                f"Test best_epoch:{best_epoch} | OA:{OA_test:.6f} | AA:{mAcc_test:.6f} | "
                f"Kappa:{Kappa_test:.6f} | mIOU:{mIOU_test:.6f}"
            )

            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
            abu_vis_path = os.path.join(save_single_experiment_folder, f"abu_test.png")
            endm_vis_path = os.path.join(save_single_experiment_folder, f"endmember_test.png")

            show_images(abu_test.detach(), save_path=abu_vis_path)
            show_endmember(end_var_test.detach().cpu().numpy(), save_path=endm_vis_path)

            for i in range(args.endm_num):
                weights_path = os.path.join(save_single_experiment_folder, f"weights_endmember_test_{i}.png")
                show_W_variants(W_logits_, b=0, p=i, save_path=weights_path)

            wandb.log({
                "test/OA": OA_test,
                "test/AA": mAcc_test,
                "test/Kappa": Kappa_test,
                "test/mIOU": mIOU_test,
                "test/prediction_map": wandb.Image(predict_save_path, caption=f"Prediction best epoch {best_epoch}"),
                "test/gt_map": wandb.Image(gt_save_path, caption="Ground truth"),
                "test/abundance_map": wandb.Image(abu_vis_path, caption=f"Abundance map best epoch {best_epoch}"),
                "test/endmember_plot": wandb.Image(endm_vis_path, caption=f"Endmember variance best epoch {best_epoch}"),
            })

        test_time = time.perf_counter() - tic_test
        Test_Time_ALL.append(test_time)

        with open(results_save_path, "a+") as f:
            str_results = (
                "\n======================"
                + f" exp_idx={exp_idx}"
                + f" seed={curr_seed}"
                + f" learning_rate={args.lr}"
                + f" query_lr={args.query_lr}"
                + f" epochs={args.max_epoch}"
                + f" train_ratio={ratio_list[0]}"
                + f" val_ratio={ratio_list[1]}"
                + " ======================"
                + f"\nOA={OA_test}"
                + f"\nAA={mAcc_test}"
                + f"\nKappa={Kappa_test}"
                + f"\nmIOU_test={mIOU_test}"
                + f"\nIOU_test={IOU_test}"
                + f"\nAcc_test={Acc_test}\n"
            )
            logger.info(str_results)
            f.write(str_results)

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        wandb.summary["final_test_OA"] = OA_test
        wandb.summary["final_test_AA"] = mAcc_test
        wandb.summary["final_test_Kappa"] = Kappa_test
        wandb.summary["train_time_sec"] = train_time
        wandb.summary["test_time_sec"] = test_time
        wandb.finish()
        del net
        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)

    logger.info(f"\n==================== Mean result of {len(seed_list)} runs ====================")
    logger.info(f"List of OA: {list(OA_ALL)}")
    logger.info(f"List of AA: {list(AA_ALL)}")
    logger.info(f"List of KPP: {list(KPP_ALL)}")
    logger.info(f"OA = {round(np.mean(OA_ALL) * 100, 2)} +- {round(np.std(OA_ALL) * 100, 2)}")
    logger.info(f"AA = {round(np.mean(AA_ALL) * 100, 2)} +- {round(np.std(AA_ALL) * 100, 2)}")
    logger.info(f"Kpp = {round(np.mean(KPP_ALL) * 100, 2)} +- {round(np.std(KPP_ALL) * 100, 2)}")
    logger.info(
        f"Acc per class = {np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)} +- "
        f"{np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)}"
    )
    logger.info(
        f"Average training time = {round(np.mean(Train_Time_ALL), 2)} +- "
        f"{round(np.std(Train_Time_ALL), 3)}"
    )
    logger.info(
        f"Average testing time = {round(np.mean(Test_Time_ALL) * 1000, 2)} +- "
        f"{round(np.std(Test_Time_ALL) * 1000, 3)}"
    )

    mean_result_path = os.path.join(save_folder, "mean_result.txt")
    with open(mean_result_path, "w") as f:
        str_results = (
            f"\n\n*************** Mean result of {len(seed_list)} runs ********************"
            + f"\nList of OA: {list(OA_ALL)}"
            + f"\nList of AA: {list(AA_ALL)}"
            + f"\nList of KPP: {list(KPP_ALL)}"
            + f"\nOA = {round(np.mean(OA_ALL) * 100, 2)} +- {round(np.std(OA_ALL) * 100, 2)}"
            + f"\nAA = {round(np.mean(AA_ALL) * 100, 2)} +- {round(np.std(AA_ALL) * 100, 2)}"
            + f"\nKpp = {round(np.mean(KPP_ALL) * 100, 2)} +- {round(np.std(KPP_ALL) * 100, 2)}"
            + f"\nAcc per class = {np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)} +- {np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)}"
            + f"\nAverage training time = {np.round(np.mean(Train_Time_ALL), 2)} +- {np.round(np.std(Train_Time_ALL), 3)}"
            + f"\nAverage testing time = {np.round(np.mean(Test_Time_ALL) * 1000, 2)} +- {np.round(np.std(Test_Time_ALL) * 1000, 3)}"
        )
        f.write(str_results)