from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loader import data_loader
from parser import get_training_parser
from utils import *
from models import CausalMotionModel
from losses import criterion
from visualize import draw_image, draw_solo, draw_solo_all
import PIL.Image
from torchvision.transforms import ToTensor
from sam.sam import SAM

from loguru import logger

@logger.catch
def main(args):
    # Set environment variables
    set_seed_globally(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    print('model name: ', get_model_name(args, time=True))
    writer = SummaryWriter(log_dir=args.tfdir + '/' + get_model_name(args, time=True), flush_secs=10)

    logging.info("Initializing Training Set")
    train_envs_path, train_envs_name = get_envs_path(args.dataset_name, "train", args.filter_envs)
    train_loaders = [data_loader(args, train_env_path, train_env_name) for train_env_path, train_env_name in
                     zip(train_envs_path, train_envs_name)]
    logging.info(train_envs_name)
    
    logging.info("Initializing Validation Set")
    val_envs_path, val_envs_name = get_envs_path(args.dataset_name, "val", args.filter_envs)#+'-'+args.filter_envs_pretrain)
    val_loaders = [data_loader(args, val_env_path, val_env_name, shuffle=False) for val_env_path, val_env_name in
                   zip(val_envs_path, val_envs_name)]
    logging.info(val_envs_name)

    # args.filter_envs_pretrain = ''

    logging.info("Initializing Validation O Set")
    val_envs_patho, val_envs_nameo = get_envs_path(args.dataset_name, "val", '0.7')
    val_loaderso = [data_loader(args, val_env_path, val_env_name) for val_env_path, val_env_name in
                   zip(val_envs_patho, val_envs_nameo)]
    logging.info(val_envs_nameo)

    ref_pictures = [[b.cuda() for b in next(iter(loader))] for loader in val_loaders]

    if args.visualize_embedding and not os.path.exists('embedding_vis/' + args.exp + '/'):
        os.makedirs('embedding_vis/' + args.exp + '/')
    
    if args.visualize_prediction and not os.path.exists('images/visualization'):
        os.makedirs('images/visualization')

    # If finetuning, we need to load the pretrain datasets for style contrastive loss
    pretrain_loaders = None
    if args.filter_envs_pretrain:
        # do we also reduce the pretrain loaders?? If yes, 
        logging.info("Initializing Training Set for contrastive loss")
        pretrain_envs_path, pretrain_envs_name = get_envs_path(args.dataset_name, "train", args.filter_envs_pretrain)
        pretrain_loaders = [data_loader(args, train_env_path, train_env_name, pt=True) for train_env_path, train_env_name in
                        zip(pretrain_envs_path, pretrain_envs_name)]
        print(pretrain_envs_name)


    # get labels of envs and create dic linking env name and env label
    # labels should be recalculated if considering left/right rule
    if ('l' in  args.filter_envs) or ('r' in args.filter_envs):
        logger.info('Using rule style...')
        def get_orientation_info(envs_name):
            style_float = []
            for d in envs_name:
                if 'True_clockwise' in  d:
                    style_float.append(1)
                elif 'False_clockwise' in d:
                    style_float.append(-1)
                else:
                    raise Exception('no orientation data in the dataset, looking for [True/False]_clockwise')
            return np.array(style_float)
        train_style_info = get_orientation_info(train_envs_name)
        val_style_info = get_orientation_info(val_envs_name)
        valo_style_info = get_orientation_info(val_envs_nameo)
        logger.info('train_enves_name {}'.format(train_envs_name))
        
        all_train_labels = sorted(set([float(d.split('_')[7]) for d in train_envs_name]*train_style_info))
        all_valid_labels = sorted(set([float(d.split('_')[7]) for d in val_envs_name]*val_style_info))
        all_valid_labelso = sorted(set([float(d.split('_')[7]) for d in val_envs_nameo]*valo_style_info))
        logger.info('all_train_labels: {}'.format(all_train_labels))
        # assert (all_train_labels == all_valid_labels)
        def get_label_name(name):
            radius = float(name.split('_')[7])
            if 'True_clockwise' in name:
                orientation = 1
            elif 'False_clockwise' in name:
                orientation = -1
            return radius * orientation
        train_labels = {name: all_train_labels.index(get_label_name(name)) for name in train_envs_name}
        val_labels = {name: all_valid_labels.index(get_label_name(name)) for name in val_envs_name}
        val_labelso = {name: all_valid_labelso.index(get_label_name(name)) for name in val_envs_nameo}
        logger.info('train_labels: {}'.format(train_labels))
        logger.info('val_labels: {}'.format(val_labels))
        logger.info('val_labelso: {}'.format(val_labels))
    else:
        all_train_labels = sorted(set([float(d.split('_')[7]) for d in train_envs_name]))
        all_valid_labels = sorted(set([float(d.split('_')[7]) for d in val_envs_name]))
        all_valid_labelso = sorted(set([float(d.split('_')[7]) for d in val_envs_nameo]))
        logger.info('all_train_labels: {}'.format(all_train_labels))
        # assert (all_train_labels == all_valid_labels)
        train_labels = {name: all_train_labels.index(float(name.split('_')[7])) for name in train_envs_name}
        val_labels = {name: all_valid_labels.index(float(name.split('_')[7])) for name in val_envs_name}
        val_labelso = {name: all_valid_labelso.index(float(name.split('_')[7])) for name in val_envs_nameo}
        logger.info('train_labels: {}'.format(train_labels))
        logger.info('val_labels: {}'.format(val_labels))
    
    if args.filter_envs_pretrain:
        all_pretrain_labels = sorted(list( set(all_train_labels) | set([float(d.split('_')[7]) for d in pretrain_envs_name])))
        pretrain_labels = {name: all_pretrain_labels.index(float(name.split('_')[7])) for name in pretrain_envs_name}
        print(all_pretrain_labels)
        print(pretrain_labels)

    
    # training routine length
    num_batches_train = min([len(train_loader) for train_loader in train_loaders])
    if args.filter_envs_pretrain: num_batches_pretrain = min([len(train_loader) for train_loader in pretrain_loaders])
    num_batches_val = min([len(val_loader) for val_loader in val_loaders])
    if val_loaderso: num_batches_valo = min([len(val_loader) for val_loader in val_loaderso]) 
    else: num_batches_valo=0

    # bring different dataset all together for simplicity of the next functions
    train_dataset = {'loaders': train_loaders, 'names': train_envs_name, 'labels': train_labels, 'num_batches': num_batches_train, 'all_train_labels': all_train_labels}
    valid_dataset = {'loaders': val_loaders, 'names': val_envs_name, 'labels': val_labels, 'num_batches': num_batches_val, 'all_valid_labels': all_valid_labels}
    valid_dataseto = {'loaders': val_loaderso, 'names': val_envs_nameo, 'labels': val_labelso, 'num_batches': num_batches_valo, 'all_valid_labelso': all_valid_labelso}
    if args.filter_envs_pretrain:
        pretrain_dataset = {'loaders': pretrain_loaders, 'names': pretrain_envs_name, 'labels': pretrain_labels, 'num_batches': num_batches_pretrain}
    else:
        pretrain_dataset = None
    

    for dataset, ds_name in zip((train_dataset, valid_dataset, pretrain_dataset), ('Train', 'Validation', 'Pretrain')):
        print(ds_name+' dataset: ', dataset)


    # create the model
    model = CausalMotionModel(args).cuda()
    
    
    
    # setup optimizer for invariant part
    if args.use_sam:
        base_optimizer = torch.optim.SGD
        inv_optimizer = SAM(model.inv_encoder.parameters(), base_optimizer, lr=args.lrstgat)
        decode_optimizer = torch.optim.Adam(
            model.decoder.parameters(),
            lr=args.lrstgat,
        )
    else:
        inv_optimizer = torch.optim.Adam(
                model.inv_encoder.parameters(),
                lr=args.lrstgat,
            )
        decode_optimizer = torch.optim.Adam(
            model.decoder.parameters(),
            lr=args.lrstgat,
        )

    # style related optimizer
    optimizers = {
        'style' : torch.optim.Adam(
            [
                {"params": model.style_encoder.encoder.parameters(), "lr": args.lrstyle},
                {"params": model.style_encoder.hat_classifier.parameters(), 'lr': args.lrclass},
            ]
        ),
        'gt_style' : torch.optim.Adam(
            [
                {"params": model.gt_encoder.encoder.parameters(), "lr": args.lrstyle},
            ]
        ),
        'inv': inv_optimizer,
        'decoder': decode_optimizer,
        'integ': (torch.optim.Adam(
            [
               {"params": model.decoder.style_blocks.parameters(), 'lr': args.lrinteg}, 
            ]    
        ) if (hasattr(model.decoder, 'style_blocks') and model.decoder.style_blocks != None) else torch.optim.Adam(model.decoder.parameters()))
    }

    if args.resume:
        load_all_model(args, model, optimizers)
        model.cuda()
    

    # training routine
    num_batches_train = min([len(train_loader) for train_loader in train_loaders]) 
    num_batches_val = min([len(val_loader) for val_loader in val_loaders]) 

    # TRAINING HAPPENS IN 6 STEPS:
    assert (len(args.num_epochs) == 6)
    # 1. (deprecated, was used for first step of stgat training)
    # 2. (deprecated, was used for second step of stgat training)
    # 3. inital training of the entire model, without any style input
    # 4. train style encoder using classifier, separate from pipeline
    # 5. train the integrator (that joins the style and the invariant features)
    # 6. fine-tune the integrator, decoder, style encoder with everything working
    training_steps = {f'P{i}': [sum(args.num_epochs[:i-1]), sum(args.num_epochs[:i])] for i in range(1, 7)}
    print(training_steps)
    def get_training_step(epoch):
        if epoch<=0: return 'P1'
        for step, r in training_steps.items():
            if r[0] < epoch <= r[1]: return step
        return 'P6'
    
    training_step = get_training_step(args.start_epoch)
    print('training step: ', training_step)
    if args.finetune:
        with torch.no_grad():
            validate_ade(model, train_dataset, args.start_epoch-1,  'P6', writer, stage='training', args=args)
            metric = validate_ade(model, valid_dataset, args.start_epoch-1,  'P6', writer, stage='validation', args=args)
            min_metric = metric
            if args.reduce == 64:
                save_all_model(args, model, optimizers, metric, -1,  'P6')
                return
            print(f'\n{"_"*150}\n')
            train_all(args, model, optimizers, train_dataset, pretrain_dataset, args.start_epoch-1, 'P6', writer, stage='training', update=False)
            train_all(args, model, optimizers, valid_dataset, pretrain_dataset, args.start_epoch-1,  'P6', writer, stage='validation')
    else:
            min_metric = 1e10

    # SOME TEST
    if args.testonly == 1:
        print('SIMPLY VALIDATE MODEL:')
        validate_ade(model, valid_dataset, 300, 'P3', writer, 'validation', write=False)
        validate_ade(model, valid_dataset, 300, 'P6', writer, 'validation', write=False)
        validate_ade(model, valid_dataseto, 300, 'P3', writer, 'validation', write=False)
        validate_ade(model, valid_dataseto, 300, 'P6', writer, 'validation', write=False)
    elif args.testonly == 2:
        print('DEPRECATED')
    elif args.testonly == 3:
        print('TEST TIME MODIF:')
        validate_ade(model, valid_dataset, 500, 'P6', writer, 'training', write=False)
        if args.ttr > 0:
            train_latent_space(args, model, valid_dataset, pretrain_dataset, writer)

    if args.testonly != 0:
        writer.close()
        return 

    
    min_metric = 1e10
    metric = min_metric
    for epoch in range(args.start_epoch, sum(args.num_epochs)+1):

        training_step = get_training_step(epoch)
        logging.info(f"\n===> EPOCH: {epoch} ({training_step})")
        
        if training_step == 'P5':
            if args.causal_decoder:
                if args.hierarchical:
                    freeze(True, (model.inv_encoder, model.style_encoder, model.decoder))
                else:
                    freeze(True, (model.inv_encoder, model.style_encoder, model.decoder.decoder))
                    freeze(False, (model.decoder.inv_norm_layer, model.decoder.style_norm_layer))
            else:
                freeze(True, (model.inv_encoder, model.style_encoder, model.decoder.mlp1, model.decoder.mlp2))
                freeze(False, (model.decoder.style_blocks,))
        elif training_step == 'P6':
            freeze(True, (model.inv_encoder,))
            freeze(False, (model.decoder, model.style_encoder))

        train_all(args, model, optimizers, train_dataset, pretrain_dataset, epoch, training_step, writer, stage='training')

        # contrastive training: train the contrastive classifier
        if args.contrastive and training_step == 'P4': model.style_encoder.train_er_classifier(train_dataset)

        #### way to decrease learning rate OPTIONAL
        # if training_step == 'P3' and epoch >= 100 and epoch % 15 == 0 or epoch >= 250 and epoch % 15 == 0:
        #     for p in optimizers['decoder'].param_groups:
        #         p['lr'] = p['lr']/2
        #         v = p['lr']
        #     for p in optimizers['inv'].param_groups:
        #         p['lr'] = p['lr']/2
        #         w = p['lr']
        #     print(f'New LRs: {v} and {w}')

        with torch.no_grad():
            if training_step == 'P4':
                if args.gt_style:
                    metric = validate_ade(model, valid_dataset, epoch, training_step, writer, stage='validation', rp=ref_pictures, args=args)
                    # validate on the training set and save to tensorboard
                    validate_ade(model, train_dataset, epoch, training_step, writer, stage='training', rp=ref_pictures, args=args)
                    validate_ade(model, valid_dataseto, epoch, training_step, writer, stage='validation_o', rp=ref_pictures, args=args)
                    
                else:
                    metric = validate_er(model, valid_dataset, epoch, writer, stage='validation')
                    validate_ade(model, valid_dataset, epoch, training_step, writer, stage='validation', rp=ref_pictures, args=args)
                    validate_ade(model, train_dataset, epoch, training_step, writer, stage='training', rp=ref_pictures, args=args)
                    validate_ade(model, valid_dataseto, epoch, training_step, writer, stage='validation_o', rp=ref_pictures, args=args)
            elif training_step in ['P3', 'P5', 'P6']:
                metric = validate_ade(model, valid_dataset, epoch, training_step, writer, stage='validation', rp=ref_pictures, args=args)
                validate_ade(model, train_dataset, epoch, training_step, writer, stage='training', rp=ref_pictures, args=args)
                validate_ade(model, valid_dataseto, epoch, training_step, writer, stage='validation_o', rp=ref_pictures, args=args)
                

            #### EVALUATE ALSO THE TRAINING ADE and the validation loss
            # validate_ade(model, valid_dataset_o, epoch, training_step, writer, stage='validation_o')
            # validate_ade(model, valid_dataseto, 300, 'P6', writer, 'validation', write=False, args=args)
            # if epoch % 2 == 0:
            #     train_all(args, model, optimizers, valid_dataset, pretrain_dataset, epoch, training_step, writer, stage='validation')                
            #     if training_step == 'P4':
            #         validate_er(model, train_dataset, epoch, writer, stage='training')
            #     else:
            #         validate_ade(model, train_dataset, epoch, training_step, writer, stage='training')
            # validate_ade(model, train_dataset, epoch, training_step, writer, stage='training', args=args)
        
        if args.finetune:
            if metric < min_metric:
                min_metric = metric
                save_all_model(args, model, optimizers, metric, epoch, training_step)
                print(f'\n{"_"*150}\n')
        else:
            if epoch % 100 == 0:
                save_all_model(args, model, optimizers, metric, epoch, training_step)
        
            
    writer.close()


def train_all(args, model, optimizers, train_dataset, pretrain_dataset, epoch, training_step, writer, stage, update=True):
    """
    Train the entire model for an epoch

    Args:
        - model (CausalMotionModel): model to train
        - optimizers: inv and style optimizers to use
        - datasets: train dataset (and pretrain dataset if finetuning)
        - stage (str): either 'validation' or 'training': says on which dataset we calculate the loss (and only backprop on 'training')
    """
    model.train()
    
    assert (stage in ['training', 'validation'])
    train_iter = [iter(loader) for loader in train_dataset['loaders']]
    pretrain_iter = [iter(loader) for loader in pretrain_dataset['loaders']] if pretrain_dataset else None
    loss_meter = AverageMeter("Loss", ":.4f")

    logging.info(f"- Computing loss ({stage})")
    tbar = tqdm(range(train_dataset['num_batches']))
    for batch_idx in tbar:
        
        # reset gradients
        for opt in optimizers.values(): opt.zero_grad()

        # compute loss (which depends on the training step)
        loss, ped_tot = criterion(model, train_iter, pretrain_iter, train_dataset, pretrain_dataset, training_step, args, stage, epoch, batch_idx)
        
        # Use SAM to train invariant part
        def closure():
            loss, _ = criterion(model, train_iter, pretrain_iter, train_dataset, pretrain_dataset, training_step, args, stage, epoch, batch_idx, go_on=False)
            loss.backward()
            return loss
        
        # backpropagate if needed
        if stage == 'training' and update:
            loss.backward()
            
            # choose which optimizer to use depending on the training step
            if args.finetune and args.finetune!='all':
                if training_step in ['P1','P2','P3',              ] and args.finetune=='stgat_enc': optimizers['inv'].step()
                if training_step in [          'P3',          'P6'] and args.finetune=='decoder'  : optimizers['decoder'].step()
                if training_step in [               'P4',     'P6'] and args.finetune in ['style', 'integ+']  : optimizers['style'].step()
                if training_step in [                    'P5','P6'] and args.finetune in ['integ', 'integ+']  : optimizers['integ'].step()
            else:
                if training_step in ['P1','P2','P3',              ]: 
                    optimizers['inv'].step(closure) if args.use_sam else optimizers['inv'].step()
                    
                if training_step in [          'P3',          'P6']: 
                    optimizers['decoder'].step()
                if training_step in [               'P4',     'P6']: 
                    if args.gt_style:
                        optimizers['inv'].step(closure) if args.use_sam else optimizers['inv'].step()
                        optimizers['decoder'].step()
                        optimizers['gt_style'].step()
                    else:
                        optimizers['style'].step()
                if training_step in [                    'P5','P6']: optimizers['integ'].step()
            
        loss_meter.update(loss.item(), ped_tot.item())
        tbar.set_description(f"Loss: {loss_meter.avg}")
    writer.add_scalar(f"{'erm' if training_step != 'P4' else 'style'}_loss/{stage}", loss_meter.avg, epoch)




def validate_ade(model, valid_dataset, epoch, training_step, writer, stage, rp=None, force=False, write=True, args=None):
    """
    Evaluate the performances on the validation set

    Args:
        - stage (str): either 'validation' or 'training': says on which dataset the metrics are computed
    """
    model.eval()

    assert (stage in ['training', 'validation', 'validation_o'])
    ade_tot_meter, fde_tot_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")
    logging.info(f"- Computing ADE ({stage})")
    with torch.no_grad():
        for loader, loader_name in zip(valid_dataset['loaders'], valid_dataset['names']):
            ade_meter, fde_meter = AverageMeter("ADE", ":.4f"), AverageMeter("FDE", ":.4f")

            for batch_idx, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, fut_traj, _, _, _, _, seq_start_end) = batch
                
                # if use ground truth model
                if args.gt_style and training_step == 'P4':
                    radius = float(loader_name.split('_')[7])
                    # clockwise is left hand traffic
                    rule = 1. if 'True_clockwise' in loader_name else -1.
                    style_embed = torch.tensor([radius, rule]).cuda()
                    pred_fut_traj_rel = model(batch, 'P4', style_embed)
                else:   
                    if training_step<='P3': ts='P3'
                    else: ts='P6'
                    pred_fut_traj_rel = model(batch, ts) if not args.visualize_embedding else model(batch, ts)[0] # from model.py the return of model(x, 'P6')

                # from relative path to absolute path
                pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])
                ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
                raw_ade = cal_ade_fde(fut_traj, pred_fut_traj, mode='sumraw')[0][1].cpu().detach().numpy() / fut_traj.shape[0]
                raw_fde = cal_ade_fde(fut_traj, pred_fut_traj, mode='sumraw')[1][1].cpu().detach().numpy()
                
                ade_, fde_ = ade_ / (obs_traj.shape[1] * fut_traj.shape[0]), fde_ / (obs_traj.shape[1])
                ade_meter.update(ade_, obs_traj.shape[1]), fde_meter.update(fde_, obs_traj.shape[1])
                ade_tot_meter.update(ade_, obs_traj.shape[1]), fde_tot_meter.update(fde_, obs_traj.shape[1])
                
                # Visualize sequence id
                max_id = np.argmax(raw_ade)
                seq_id = max_id # 0
                
                # Visualize first batch every 10 epochs of OOD performance
                if args.visualize_prediction and batch_idx in [0, 2, 4] and (epoch % 10 == 0) and (stage in ['validation_o', 'validation']):
                # visualize output
                    idx_start, idx_end = seq_start_end[seq_id//2][0], seq_start_end[seq_id//2][1]
                    obsv_scene = obs_traj[:, idx_start:idx_end, :]
                    pred_scene = pred_fut_traj[:, idx_start:idx_end, :]
                    gt_scene = fut_traj[:, idx_start:idx_end, :]
                    scene_name = loader_name.split('_')[7] + ('cw' if 'True_clockwise' in loader_name else 'ccw')
                    # compute ADE and FDE metrics
                    if not os.path.exists('./images/visualization/{}_{}'.format(stage, args.exp)):
                        os.makedirs('./images/visualization/{}_{}'.format(stage, args.exp))
                    figname = './images/visualization/{}_{}/{}_epoch{}_{}_seq_{:02d}_{:02d}_sample_ade{:.3f}_fde{:.3f}.pdf'.format(
                        stage, args.exp, scene_name, epoch, loader_name, seq_id, batch_idx, raw_ade[seq_id], raw_fde[seq_id])
                    bar_figname = './images/visualization/{}_{}/{}_epoch{}_{}_{:02d}_mean_ade{:.3f}_fde{:.3f}_bar.pdf'.format(
                        stage, args.exp, scene_name, epoch, loader_name, batch_idx, ade_, fde_)
                    bartitle = '{} mean_ADE: {:.3f} | FDE:{:.3f}'.format(
                        scene_name, ade_, fde_)
                    bar_buf = plotbar(
                        raw_ade, raw_fde, figname=bar_figname, epoch=epoch, title=bartitle)
                    figtitle = '{} Epoch {} Seq{} ADE{:.3f}  FDE{:.3f}'.format(
                        scene_name, epoch, seq_id, raw_ade[seq_id], raw_fde[seq_id])
                    scene_buf = sceneplot(obsv_scene.permute(1, 0, 2).cpu().detach().numpy(), pred_scene.permute(1, 0, 2).cpu().detach().numpy(), 
                                gt_scene.permute(1, 0, 2).cpu().detach().numpy(), figname=figname, title=figtitle)
                    bar_img = PIL.Image.open(bar_buf)
                    bar_img = ToTensor()(bar_img)
                    if write:
                        writer.add_image('{}: Bar Image of {}'.format(scene_name, batch_idx), bar_img, epoch)
                        scene_img = PIL.Image.open(scene_buf)
                        scene_img = ToTensor()(scene_img)
                        writer.add_image('{}: Scene Image of {}'.format(scene_name, batch_idx), scene_img, epoch)

            logging.info(f'\t\t ADE on {loader_name:<25} dataset:\t {ade_meter.avg}')

    logging.info(f"Average {stage}:\tADE  {ade_tot_meter.avg:.4f}\tFDE  {fde_tot_meter.avg:.4f}")
    # repoch = epoch if args.num_epochs[4] == 0 else epoch - 370
    repoch = epoch
    if write: 
        writer.add_scalar(f"ade/{stage}", ade_tot_meter.avg, repoch)
        writer.add_scalar(f"fde/{stage}", fde_tot_meter.avg, repoch)

    ## SAVE VISUALIZATIONS
    # if epoch % 1 == 0 and stage == 'validation':
    # if (stage == 'validation' and rp != None and epoch % 3 == 0) or force and write:
        
    #     obs = [b[0] for b in rp]
    #     fut = [b[1] for b in rp]
    #     pred = [relative_to_abs(model(b, ts), b[0][-1, :, :2]) for b in rp]
    #     res = [[obs[i], fut[i], pred[i]] for i in range(len(rp))]
    #     fig, array = draw_image(res)
    #     fig.savefig(f'images/visu/pred{epoch}.png')
    #     writer.add_image("Some paths", array, epoch)

    return ade_tot_meter.avg


def validate_er(model, valid_dataset, epoch, writer, stage):
    """
    Evaluate the performances on the validation set

    Args:
        - stage (str): either 'validation' or 'training': says on which dataset the metrics are computed
    """
    model.eval()

    assert (stage in ['training', 'validation'])
    er_tot_meter = AverageMeter('error_rate_tot', ":.4f")

    logging.info(f"- Computing style error_rate ({stage})")
    with torch.no_grad():
        for loader, loader_name in zip(valid_dataset['loaders'], valid_dataset['names']):
            er_meter = AverageMeter('error_rate', ":.4f")
            label = valid_dataset['labels'][loader_name]

            for batch in loader:
                batch = [tensor.cuda() for tensor in batch]
                output = model(batch, 'P4')
                label_preds = output.argmax(dim=-1)

                metric = ((label_preds != label) * 1).to(float).mean()
                er_meter.update(metric.item()), er_tot_meter.update(metric.item())

            logging.info(f'\t\t error_rate on {loader_name:<25} dataset:\t {er_meter.avg}')
        logging.info(f"average {stage} error_rate:\t  {er_tot_meter.avg:.4f}")
        writer.add_scalar(f"error_rate/{stage}", er_tot_meter.avg, epoch)
    return er_tot_meter.avg


def train_latent_space(args, model, train_dataset, pretrain_dataset, writer):

    freeze(True, (model,)) # freeze all models, it's test time
    model.eval()
    ade_tot_meters = [AverageMeter('error_rate_tot', ":.4f") for _ in range(args.ttr+1)]
    loss_tot_meters = [AverageMeter('loss_meter', ":.4f") for _ in range(args.ttr+1)]

    logging.info(f"- Optimizing latent spaces ")
    
    for loader, loader_name in zip(train_dataset['loaders'], train_dataset['names']):
        label = train_dataset['labels'][loader_name]

        for idx, batch in enumerate(loader):

            # get this batch
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, fut_traj, _, _, _, style_input, _) = batch

            # fig, array = draw_image([[obs_traj, fut_traj, fut_traj]])
            # writer.add_image("RefImages", array, 0)
            # return

            # encode the latent spaces that we'll optimize, create optimizer
            inv_latent_space = model.inv_encoder(obs_traj)
            inv_latent_space = inv_latent_space.detach()
            inv_latent_space.requires_grad = True
            opt = torch.optim.Adam((inv_latent_space,), lr=args.ttrlr)

            # encode the ground truth style that we'll used as goal
            ref_low_dim, style_encoding = model.style_encoder(style_input, 'both')
            ref_low_dim = ref_low_dim.detach()
            
            if args.wrongstyle:
                print(pretrain_dataset['names'])
                other_style_input = next(iter(pretrain_dataset['loaders'][2]))[5].cuda()
                other_ref_low_dim, _ = model.style_encoder(other_style_input, 'both')
                other_ref_low_dim = other_ref_low_dim.detach()
                lab_tensor = torch.stack((torch.tensor(label).cuda(),torch.tensor(label+1).cuda()), dim=0)
            else:
                lab_tensor = torch.tensor(label).cuda().unsqueeze(0)
            
            wot_num = 64
            evolutions = [[] for _ in range(wot_num)] # store the predictions at each step for visualization

            for wto in tqdm(range(wot_num)): # we optimize seq per seq. ID of the seq we'll optimize
                
                for k in range(args.ttr): # number of steps of optimization

                    opt.zero_grad()
                    
                    # do the prediction, compute the low dim style space of the prediction
                    traj_pred_rel_k = model.decoder(inv_latent_space, style_encoding)
                    traj_pred_k = relative_to_abs(traj_pred_rel_k, obs_traj[-1, :, :2])
                    pred_full_path = torch.cat((obs_traj, traj_pred_k))
                    pred_style = model.style_encoder(from_abs_to_social(pred_full_path), 'low')

                    if args.wrongstyle:
                        # set label to wrong label to harm the prediction
                        other_style = torch.clone(other_ref_low_dim)
                        other_style[wto] = pred_style[wto]
                        style_tensor = torch.stack((torch.clone(ref_low_dim), other_style), dim=0) # get the batch of social encoding of sequences of ONE ENV
                    
                    else:
                        # replace the first seq style GT by first seq style prediction
                        style_tensor = torch.clone(ref_low_dim).unsqueeze(0) # get the batch of social encoding of sequences of ONE ENV
                        style_tensor[0][wto] = pred_style[wto] # replace seq number WTO in the batch of seq social encodings
                    
                    # compute loss           
                    loss = criterion.contrastive_loss(style_tensor, lab_tensor)

                    # update metrics & visualization
                    loss_tot_meters[k].update(loss.item())
                    # ade_list.append(compute_ade_single(traj_pred_rel_k, obs_traj, fut_traj, wto))
                    ade_tot_meters[k].update(compute_ade_single(traj_pred_rel_k, obs_traj, fut_traj, wto)) # compute_ade_single() compute ADE just on seq number WTO 
                    if k in [0, 1, 5, 9] + [i*20-1 for i in range(20)]:
                        fig, array = draw_image([[obs_traj, fut_traj, traj_pred_k.detach()]])
                        writer.add_image("Some paths", array, k)
                    evolutions[wto].append(traj_pred_k.detach()) # save for visualization

                    # backward and optimize
                    loss.backward()
                    opt.step()

                traj_pred_rel_k = model.decoder(inv_latent_space, style_encoding)
                ade_tot_meters[k+1].update(compute_ade_single(traj_pred_rel_k, obs_traj, fut_traj, wto))
                evolutions[wto].append(traj_pred_k.detach())

    all_res = []
    for evo in evolutions:
        res = [[obs_traj, fut_traj, pred] for i, pred in enumerate(evo) if i in [0, 1, 3, 5, 10]]
        all_res.append(res)
    
    fig, array = draw_solo_all(all_res)
    writer.add_image("evol/refinement", array, 0)

    for k in range(args.ttr+1):
        logging.info(f"average ade during refinement [{k}]:\t  {ade_tot_meters[k].avg:.6f}   \t  loss refinement [{k}]:\t  {loss_tot_meters[k].avg:.8f} ")
        writer.add_scalar(f"ade_refine/plot", ade_tot_meters[k].avg, k)
        if k < args.ttr: writer.add_scalar(f"loss_refine/plot", loss_tot_meters[k].avg, k)


def compute_ade_(pred_fut_traj_rel, obs_traj, fut_traj):
    pred_fut_traj = relative_to_abs(pred_fut_traj_rel, obs_traj[-1, :, :2])
    ade_, fde_ = cal_ade_fde(fut_traj, pred_fut_traj)
    ade_ = ade_ / (fut_traj.shape[0] * obs_traj.shape[1])
    return ade_

def compute_ade_single(pred_fut_traj_rel, obs_traj, fut_traj, wto):
    return compute_ade_(pred_fut_traj_rel[:, wto*NUMBER_PERSONS:NUMBER_PERSONS*(wto+1)], obs_traj[:, wto*NUMBER_PERSONS:NUMBER_PERSONS*(wto+1)], fut_traj[:, wto*NUMBER_PERSONS:NUMBER_PERSONS*(wto+1)])


if __name__ == "__main__":
    print('Using GPU: ' + str(torch.cuda.is_available()))
    input_args = get_training_parser().parse_args()
    print('Arguments for training: ', input_args)
    set_logger(os.path.join(input_args.log_dir, "train.log"))
    main(input_args)