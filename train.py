import os
import yaml
import shutil
import time
import argparse
import torch.nn as nn
import logging as log

from pathlib import Path
from criteria import *
from dataloader import BatchDataLoader, SpeechMixDataset
from utils.Checkpoint import Checkpoint
from networks.speakerfilter import NetLstm
from utils.progressbar import progressbar as pb
from utils.util import makedirs, saveYAML
import numpy as np
import random
def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False


def validate(network, eval_loader, weight, *criterion):
    network.eval()
    with torch.no_grad():
        cnt = 0.
        accu_eval_loss = 0.0
        ebar = pb(0, len(eval_loader.get_dataloader()), 20)
        ebar.start()
        for j, batch_eval in enumerate(eval_loader.get_dataloader()):
            mixture, anchor = batch_info[0].cuda(), batch_info[2].cuda()
            outputs = network(mixture,anchor)
            loss = 0.
            for idx, cri in enumerate(criterion):
                loss += cri(outputs, batch_eval) * weight[idx]
            eval_loss = loss.data.item()
            accu_eval_loss += eval_loss
            cnt += 1.
            ebar.update_progress(j, 'CV   ', 'loss:{:.5f}/{:.5f}'.format(eval_loss, accu_eval_loss / cnt))

        avg_eval_loss = accu_eval_loss / cnt
    print()
    network.train()
    return avg_eval_loss


if __name__ == '__main__':
    setup_seed(475)
    """
    environment part
    """
    # loading argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-y", "--yaml_name", help="config file name")
    args = parser.parse_args()

    # loading config
    _abspath = Path(os.path.abspath(__file__)).parent
    _project = _abspath.stem
    _yaml_path = os.path.join(_abspath, 'configs/' + args.yaml_name)
    try:
        with open(_yaml_path, 'r') as f_yaml:
            config = yaml.load(f_yaml, Loader=yaml.FullLoader)
    except:
        raise ValueError('No config file found at "%s"' % _yaml_path)

    # make output dirs
    _outpath = config['OUTPUT_DIR'] + _project + config['WORKSPACE']
    _modeldir = _outpath + '/checkpoints/'
    _datadir = _outpath + '/estimations/'
    _logdir = _outpath + '/log/'
    makedirs([_modeldir, _datadir, _logdir])
    saveYAML(config, _outpath + '/' + args.yaml_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_ID']

    """
    network part
    """
    # dataset
    tr_mix_dataset = SpeechMixDataset(config,os.path.join(config['DATA_PATH'],'tr/'))
    tr_batch_dataloader = BatchDataLoader(tr_mix_dataset, config['BATCH_SIZE'], is_shuffle=True,
                                          workers_num=config['NUM_WORK'])
    if config['USE_CV']:
        cv_mix_dataset = SpeechMixDataset(config, os.path.join(config['DATA_PATH'],'cv/'))
        cv_batch_dataloader = BatchDataLoader(cv_mix_dataset, config['BATCH_SIZE'], is_shuffle=False,
                                              workers_num=config['NUM_WORK'])

    # device setting

    # set model and optimizer
    network = NetLstm()
    network.cuda()
    parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("Trainable parameters : " + str(parameters))
    optimizer = torch.optim.Adam(network.parameters(), lr=config['LR'], amsgrad=True)
    lr_list = [0.0002] * 50 + [0.0001] * 50 + [0.00005] * 50 + [0.00001] * 50
    #  criteria,weight for each criterion
    criterion = stftm_loss(config['WIN_LEN'], config['WIN_OFFSET'], loss_type='mse')
    weight = [1.]

    if args.model_name == 'none':
        log.info('#' * 12 + 'NO EXIST MODEL, TRAIN NEW MODEL ' + '#' * 12)
        best_loss = float('inf')
        start_epoch = 0
    else:
        checkpoint = Checkpoint()
        checkpoint.load(args.model_name)
        start_epoch = checkpoint.start_epoch
        best_loss = checkpoint.best_loss
        network.load_state_dict(checkpoint.state_dict)
        optimizer.load_state_dict(checkpoint.optimizer)
        log.info('#' * 18 + 'Finish Resume Model ' + '#' * 18)

    """
    training part
    """
    log.info('#' * 20 + ' START TRAINING ' + '#' * 20)
    cnt = 0.  #
    for epoch in range(start_epoch, config['MAX_EPOCH']):
        # set learning rate for every epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_list[epoch]

        # initial param
        accu_train_loss = 0.0
        network.train()
        tbar = pb(0, len(tr_batch_dataloader.get_dataloader()), 20)
        tbar.start()

        for i, batch_info in enumerate(tr_batch_dataloader.get_dataloader()):
            mixture, anchor = batch_info[0].cuda(), batch_info[2].cuda()

            # forward + backward + optimize
            optimizer.zero_grad()

            outputs = network(mixture,anchor)
            loss = criterion(outputs, batch_info)
            loss.backward()
            optimizer.step()

            # calculate losses
            running_loss = loss.data.item()
            accu_train_loss += running_loss

            # display param
            cnt += 1
            del loss, outputs, batch_info

            tbar.update_progress(i, 'Train',
                                 'epoch:{}/{}, lr:{}, loss:{:.5f}/{:.5f}'.format(epoch + 1, config['MAX_EPOCH'],
                                                                                 param_group['lr'],
                                                                                 running_loss, accu_train_loss / cnt))
            if config['USE_CV'] and (i + 1) % config['EVAL_STEP'] == 0:
                print()
                avg_train_loss = accu_train_loss / cnt
                avg_eval_loss = validate(network, cv_batch_dataloader, weight, criterion)
                is_best = True if avg_eval_loss < best_loss else False
                best_loss = avg_eval_loss if is_best else best_loss
                log.info('Epoch [%d/%d], ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
                    epoch + 1, config['MAX_EPOCH'], avg_train_loss, avg_eval_loss))

                checkpoint = Checkpoint(epoch + 1, avg_train_loss, best_loss, network.state_dict(),
                                        optimizer.state_dict())
                model_name = _modeldir + '{}-{}-val.ckpt'.format(epoch + 1, i + 1)
                best_model = _modeldir + 'best.ckpt'
                if is_best:
                    checkpoint.save(is_best, best_model)
                if not config['SAVE_BEST_ONLY']:
                    checkpoint.save(False, model_name)

                accu_train_loss = 0.0
                network.train()
                cnt = 0.


    timeit = time.strftime('%Y-%m-%d-%H_', time.localtime(time.time()))
    log_path = str(_abspath) + '/train.log'
    if os.path.exists(log_path):
        shutil.copy(log_path, _outpath + '/log/' + timeit + 'train.log')
        file = open(log_path, 'w').close()
