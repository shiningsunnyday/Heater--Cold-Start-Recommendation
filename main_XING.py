import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn import datasets
import data
import model

import argparse
import pickle
from sklearn import preprocessing as prep
from tqdm import tqdm
import scipy.sparse
import json
from copy import deepcopy
np.random.seed(0)
tf.set_random_seed(0)


def main():
    data_name = args.data
    model_select = args.model_select
    rank_out = args.rank
    data_batch_size = 1024
    dropout = args.dropout
    recall_at = [20, 50, 100]
    eval_batch_size = 5000  # the batch size when test
    eval_every = args.eval_every
    num_epoch = 100
    neg = args.neg

    _lr = args.lr
    _decay_lr_every = 3
    _lr_decay = 0.8

    if data_name == "anime_cold":
        dat = load_data_anime_cold() 
    elif data_name == "anime_warm":
        dat = load_data_anime_warm()
    else:
        dat = load_data(data_name)
    test_item_eval = dat['test_item_eval']
    test_user_eval = dat['test_user_eval']
    test_user_item_eval = dat['test_user_item_eval']
    vali_item_eval = dat['vali_item_eval']
    vali_user_eval = dat['vali_user_eval']
    vali_user_item_eval = dat['vali_user_item_eval']
    user_content = dat['user_content']
    item_content = dat['item_content']
    u_pref = dat['u_pref']
    v_pref = dat['v_pref']
    user_list = dat['user_list']
    item_list = dat['item_list']
    item_warm = np.unique(item_list)
    user_warm = np.unique(user_list)

    timer = utils.timer(name='main').tic()

    # prep eval
    eval_batch_size = eval_batch_size
    timer.tic()
    test_item_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                                cold_user=True, cold_item=True)
    test_user_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                                cold_user=True, cold_item=True)
    test_user_item_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                                cold_user=True, cold_item=True)
    vali_item_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                                cold_user=True, cold_item=True)
    vali_user_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                                cold_user=True, cold_item=True)
    vali_user_item_eval.init_tf(u_pref, v_pref, user_content, item_content, eval_batch_size,
                                cold_user=True, cold_item=True)

    heater = model.Heater(latent_rank_in=u_pref.shape[1],
                               user_content_rank=user_content.shape[1],
                               item_content_rank=item_content.shape[1],
                               model_select=model_select,
                               rank_out=rank_out, reg=args.reg, alpha=args.alpha, dim=args.dim)

    config = tf.ConfigProto(allow_soft_placement=True)

    heater.build_model()
    heater.build_predictor(recall_at)

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        timer.toc('initialized tf')

        n_step = 0
        best_recall_item = [0, 0, 0]
        best_recall_user = [0, 0, 0]
        best_recall_user_item = [0, 0, 0]
        best_test_item_recall = [0, 0, 0]
        best_test_user_recall = [0, 0, 0]
        best_test_user_item_recall = [0, 0, 0]
        best_step = 0
        tf.local_variables_initializer().run()
        for epoch in range(num_epoch):
            user_array, item_array, target_array = utils.negative_sampling(user_list, item_list, neg, item_warm)
            random_idx = np.random.permutation(user_array.shape[0])
            n_targets = len(random_idx)
            data_batch = [(n, min(n + data_batch_size, n_targets)) for n in range(0, n_targets, data_batch_size)]
            loss_epoch = 0.
            reg_loss_epoch = 0.
            diff_loss_epoch = 0.
            expert_loss_epoch = 0.
            gen = data_batch
            gen = tqdm(gen)
            for (start, stop) in gen:
                n_step += 1

                batch_idx = random_idx[start:stop]
                batch_users = user_array[batch_idx]
                batch_items = item_array[batch_idx]
                batch_targets = target_array[batch_idx]

                # dropout
                if dropout != 0:
                    n_to_drop = int(np.floor(dropout * len(batch_idx)))  # number of u-i pairs to be dropped
                    zero_item_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                    zero_user_index = np.random.choice(np.arange(len(batch_idx)), n_to_drop, replace=False)
                else:
                    zero_item_index = np.array([])
                    zero_user_index = np.array([])

                item_content_batch = item_content[batch_items, :]
                user_content_batch = user_content[batch_users, :]
                dropout_item_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                if len(zero_item_index) > 0:
                    dropout_item_indicator[zero_item_index] = 1
                dropout_user_indicator = np.zeros_like(batch_targets).reshape((-1, 1))
                if len(zero_user_index) > 0:
                    dropout_user_indicator[zero_user_index] = 1

                _, _, loss_out, reg_loss_out, diff_loss_out = sess.run(
                    [heater.preds, heater.optimizer, heater.loss,
                     heater.reg_loss, heater.diff_loss],
                    feed_dict={
                        heater.Uin: u_pref[batch_users, :],
                        heater.Vin: v_pref[batch_items, :],
                        heater.Ucontent: user_content_batch,
                        heater.Vcontent: item_content_batch,
                        heater.dropout_user_indicator: dropout_user_indicator,
                        heater.dropout_item_indicator: dropout_item_indicator,
                        heater.target: batch_targets,
                        heater.lr_placeholder: _lr,
                        heater.is_training: True
                    }
                )
                loss_epoch += loss_out
                reg_loss_epoch += reg_loss_out
                diff_loss_epoch += diff_loss_out
                if np.isnan(loss_epoch):
                    raise Exception('f is nan')

            if (epoch + 1) % _decay_lr_every == 0:
                _lr = _lr_decay * _lr
                print('decayed lr:' + str(_lr))

            if epoch % eval_every == 0:
                recall_vali_item, precision_vali_item, \
                ndcg_vali_item = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                         eval_feed_dict=heater.get_eval_dict,
                                                         recall_k=recall_at, eval_data=vali_item_eval)
                print('vali item done!')
                recall_vali_user, precision_vali_user, \
                ndcg_vali_user = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                         eval_feed_dict=heater.get_eval_dict,
                                                         recall_k=recall_at, eval_data=vali_user_eval)
                print('vali user done!')
                recall_vali_user_item, precision_vali_user_item, \
                ndcg_vali_user_item = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                              eval_feed_dict=heater.get_eval_dict,
                                                              recall_k=recall_at, eval_data=vali_user_item_eval)
                print('vali user-item done!')

            is_best = False
            if (np.sum(recall_vali_item)
                + np.sum(recall_vali_user)
                + np.sum(recall_vali_user_item)) > (np.sum(best_recall_item)
                                                    + np.sum(best_recall_user)
                                                    + np.sum(best_recall_user_item)):
                is_best = True

                best_recall_item = recall_vali_item
                best_recall_user = recall_vali_user
                best_recall_user_item = recall_vali_user_item

                recall_test_item, precision_test_item, \
                ndcg_test_item = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                         eval_feed_dict=heater.get_eval_dict,
                                                         recall_k=recall_at, eval_data=test_item_eval)
                print('test item done!')
                recall_test_user, precision_test_user, \
                ndcg_test_user = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                         eval_feed_dict=heater.get_eval_dict,
                                                         recall_k=recall_at, eval_data=test_user_eval)
                print('test user done!')
                recall_test_user_item, precision_test_user_item, \
                ndcg_test_user_item = utils.batch_eval_recall(sess, heater.eval_preds_cold,
                                                              eval_feed_dict=heater.get_eval_dict,
                                                              recall_k=recall_at, eval_data=test_user_item_eval)
                print('test user-item done!')

                best_test_item_recall = recall_test_item
                best_test_user_recall = recall_test_user
                best_test_user_item_recall = recall_test_user_item
                best_epoch = epoch

            timer.toc('%d [%d]b loss=%.4f reg_loss=%.4f diff_loss=%.4f expert_loss=%.4f best[%d]' % (
                epoch, len(data_batch), loss_epoch, reg_loss_epoch, diff_loss_epoch, expert_loss_epoch, best_step
            )).tic()
            print('\t\t\t' + '\t '.join([('@' + str(i)).ljust(6) for i in recall_at]))
            print('Current item recall\t\t%s' % (
                ' '.join(['%.6f' % i for i in recall_vali_item]),
            ))
            print('Current item precision\t%s' % (
                ' '.join(['%.6f' % i for i in precision_vali_item]),
            ))
            print('Current item ndcg\t\t%s' % (
                ' '.join(['%.6f' % i for i in ndcg_vali_item]),
            ))

            print('-' * 30)

            print('Current user recall\t\t%s' % (
                ' '.join(['%.6f' % i for i in recall_vali_user]),
            ))
            print('Current user precision\t%s' % (
                ' '.join(['%.6f' % i for i in precision_vali_user]),
            ))
            print('Current user ndcg\t\t%s' % (
                ' '.join(['%.6f' % i for i in ndcg_vali_user]),
            ))

            print('-' * 30)

            print('Current user-item recall\t\t%s' % (
                ' '.join(['%.6f' % i for i in recall_vali_user_item]),
            ))
            print('Current user-item precision\t%s' % (
                ' '.join(['%.6f' % i for i in precision_vali_user_item]),
            ))
            print('Current user-item ndcg\t\t%s' % (
                ' '.join(['%.6f' % i for i in ndcg_vali_user_item]),
            ))

            if is_best:
                print('=' * 50)

                print('Current test item recall\t\t%s' % (
                    ' '.join(['%.6f' % i for i in recall_test_item]),
                ))
                print('Current test item precision\t%s' % (
                    ' '.join(['%.6f' % i for i in precision_test_item]),
                ))
                print('Current test item ndcg\t\t%s' % (
                    ' '.join(['%.6f' % i for i in ndcg_test_item]),
                ))

                print('-' * 30)

                print('Current test user recall\t\t%s' % (
                    ' '.join(['%.6f' % i for i in recall_test_user]),
                ))
                print('Current test user precision\t%s' % (
                    ' '.join(['%.6f' % i for i in precision_test_user]),
                ))
                print('Current test user ndcg\t\t%s' % (
                    ' '.join(['%.6f' % i for i in ndcg_test_user]),
                ))

                print('-' * 30)

                print('Current test user-item recall\t\t%s' % (
                    ' '.join(['%.6f' % i for i in recall_test_user_item]),
                ))
                print('Current test user-item precision\t%s' % (
                    ' '.join(['%.6f' % i for i in precision_test_user_item]),
                ))
                print('Current test user-item ndcg\t\t%s' % (
                    ' '.join(['%.6f' % i for i in ndcg_test_user_item]),
                ))

            print('=' * 50)

            print('best epoch[%d]\t vali item recall: %s' % (
                best_epoch,
                ' '.join(['%.6f' % i for i in best_recall_item]),
            ))
            print('best epoch[%d]\t vali user recall: %s' % (
                best_epoch,
                ' '.join(['%.6f' % i for i in best_recall_user]),
            ))
            print('best epoch[%d]\t vali user-item recall: %s' % (
                best_epoch,
                ' '.join(['%.6f' % i for i in best_recall_user_item]),
            ))
            print('best epoch[%d]\t test item recall: %s' % (
                best_epoch,
                ' '.join(['%.6f' % i for i in best_test_item_recall]),
            ))
            print('best epoch[%d]\t test user recall: %s' % (
                best_epoch,
                ' '.join(['%.6f' % i for i in best_test_user_recall]),
            ))
            print('best epoch[%d]\t test user-item recall: %s' % (
                best_epoch,
                ' '.join(['%.6f' % i for i in best_test_user_item_recall]),
            ))


ANIME_PATH="/dfs/user/msun415/anime/data/anime/train_val_test/10000"


def prep_standardize_dense(x):
    scaler = prep.StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    x_scaled[x_scaled > 5] = 5
    x_scaled[x_scaled < -5] = -5
    x_scaled[np.absolute(x_scaled) < 1e-5] = 0
    return scaler, x_scaled

def load_data_anime_cold():
    VAR_NAMES=['u_feats','v_feats','val_v_feats']
    for j in VAR_NAMES: 
        dic=json.load(open(f"/dfs/user/msun415/anime/cached/anime2vec/bert-pretrained/{j}.json"))
        globals()[j]={int(float(k)):np.array(dic[k]) for k in dic}

    v_feats_comb = deepcopy(v_feats)
    v_feats_comb.update(val_v_feats)
    
    user_path="{}/10000_train_user_factors.csv".format(ANIME_PATH)
    item_path="{}/10000_train_item_factors.csv".format(ANIME_PATH)
    pref_path="{}/10000_train_pref_matrix.csv".format(ANIME_PATH)
    val_pref_path="{}/10000_val_pref_matrix.csv".format(ANIME_PATH)
    pref_matrix=pd.read_csv(pref_path).iloc[:,1:]
    val_pref_matrix=pd.read_csv(val_pref_path).iloc[:,1:]
    pref_mask=pref_matrix>0.0
    val_pref_mask=val_pref_matrix>0.0

    user_df=pd.read_csv(user_path)    
    item_df=pd.read_csv(item_path)    
    user_vectors=pd.read_csv(user_path).iloc[:,1:]
    item_vectors=pd.read_csv(item_path).iloc[:,1:]
    user_ids=user_df.iloc[:,0]#inds
    train_ids=list(v_feats.keys())
    val_ids=list(val_v_feats.keys())
    u,v=user_vectors.values,item_vectors.values
    _,u=prep_standardize_dense(u)
    _,v=prep_standardize_dense(v)
    u_dict=dict(zip(user_ids,u))#{user_ind:item_vec}
    v_dict=dict(zip(train_ids,v))#{anime_ind:item_vec}
    val_v_dict=dict(zip(val_ids, np.zeros((len(val_ids), 200))))
    v_dict_comb = deepcopy(v_dict)
    v_dict_comb.update(val_v_dict)

    dat = {}
    dat['u_pref'] = u
    dat['v_pref'] = np.array(list(v_dict_comb.values()))   
    dat['item_content'] = np.array(list(v_feats_comb.values()))
    dat['user_content'] = np.array(list(u_feats.values()))
    
    pd.DataFrame(np.argwhere(pref_mask.values==1),columns=['uid','iid']).to_csv(f"{ANIME_PATH}/train.csv",index=False)
    pd.DataFrame(np.argwhere(val_pref_mask.values==1),columns=['uid','iid']).to_csv(f"{ANIME_PATH}/test.csv",index=False)
    train = pd.read_csv(f"{ANIME_PATH}/train.csv", dtype=np.int32)
    dat['user_list'] = train['uid'].values
    dat['item_list'] = train['iid'].values
    dat['vali_item_eval'] = data.load_eval_data(f"{ANIME_PATH}/test.csv")
    dat['vali_user_eval'] = data.load_eval_data(f"{ANIME_PATH}/test.csv")
    dat['test_item_eval'] = data.load_eval_data(f"{ANIME_PATH}/test.csv")
    dat['test_user_eval'] = data.load_eval_data(f"{ANIME_PATH}/test.csv")
    dat['test_user_item_eval'] = data.load_eval_data(f"{ANIME_PATH}/test.csv")    
    dat['vali_user_item_eval'] = data.load_eval_data(f"{ANIME_PATH}/test.csv")    
    return dat


def load_data(data_name):
    timer = utils.timer(name='main').tic()
    data_path = './data/' + data_name
    u_file = data_path + '/U_BPR.npy'
    v_file = data_path + '/V_BPR.npy'
    user_content_file = data_path + '/user_content.npz'
    item_content_file = data_path + '/item_content.npz'
    train_file = data_path + '/train.csv'
    vali_item_file = data_path + '/vali_item.csv'
    vali_user_file = data_path + '/vali_user.csv'
    vali_user_item_file = data_path + '/vali_user_item.csv'
    test_item_file = data_path + '/test_item.csv'
    test_user_file = data_path + '/test_user.csv'
    test_user_item_file = data_path + '/test_user_item.csv'
    with open('./data/' + data_name + '/info.pkl', 'rb') as f:
        info = pickle.load(f)
        num_user = info['num_user']
        num_item = info['num_item']

    dat = {}
    # load preference data
    timer.tic()
    u_pref = np.load(u_file)
    v_pref = np.load(v_file)
    dat['u_pref'] = u_pref
    dat['v_pref'] = v_pref

    timer.toc('loaded U:%s,V:%s' % (str(u_pref.shape), str(v_pref.shape))).tic()

    # pre-process
    _, dat['u_pref'] = utils.standardize(u_pref)
    _, dat['v_pref'] = utils.standardize_3(v_pref)
    timer.toc('standardized U,V').tic()

    # load content data
    timer.tic()
    user_content = scipy.sparse.load_npz(user_content_file)
    dat['user_content'] = user_content.tolil(copy=False)
    item_content = scipy.sparse.load_npz(item_content_file)
    dat['item_content'] = item_content.tolil(copy=False)
    timer.toc('loaded item feature sparse matrix: %s' % (str(item_content.shape))).tic()

    # load split
    timer.tic()
    train = pd.read_csv(train_file, dtype=np.int32)
    dat['user_list'] = train['uid'].values
    dat['item_list'] = train['iid'].values
    dat['warm_item'] = np.unique(train['iid'].values)
    timer.toc('read train triplets %s' % str(train.shape)).tic()

    dat['test_item_eval'] = data.load_eval_data(test_item_file)
    dat['test_user_eval'] = data.load_eval_data(test_user_file, cold_user=True, test_item_ids=dat['warm_item'])
    dat['test_user_item_eval'] = data.load_eval_data(test_user_item_file)
    dat['vali_item_eval'] = data.load_eval_data(vali_item_file)
    dat['vali_user_eval'] = data.load_eval_data(vali_user_file, cold_user=True, test_item_ids=dat['warm_item'])
    dat['vali_user_item_eval'] = data.load_eval_data(vali_user_item_file)
    breakpoint()
    return dat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main_XING",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', type=str, default='XING', help='path to eval in the downloaded folder')
    parser.add_argument('--model-select', nargs='+', type=int,
                        default=[200],
                        help='specify the fully-connected architecture, starting from input,'
                             ' numbers indicate numbers of hidden units')
    parser.add_argument('--rank', type=int, default=200, help='output rank of latent model')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--eval-every', type=int, default=1, help='evaluate every X user-batch')
    parser.add_argument('--neg', type=float, default=5, help='negative sampling rate')
    parser.add_argument('--lr', type=float, default=0.005, help='starting learning rate')
    parser.add_argument('--alpha', type=float, default=0.0001, help='diff loss parameter')
    parser.add_argument('--reg', type=float, default=0.0001, help='regularization')
    parser.add_argument('--dim', type=int, default=5, help='number of experts')

    args = parser.parse_args()
    args, _ = parser.parse_known_args()
    for key in vars(args):
        print(key + ":" + str(vars(args)[key]))
    main()
