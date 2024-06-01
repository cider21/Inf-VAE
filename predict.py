import operator

import tensorflow as tf
import numpy as np
import sys

from models.infvae_models import InfVAESocial, InfVAECascades
from utils.preprocess import *
from utils.flags import *

def predict(session, model, feed):
    """ Helper function to compute model predictions. """
    recall_scores, map_scores, n_samples, top_k, target = \
        session.run([model.recall_scores, model.map_scores, model.relevance_scores,
                     model.top_k_filter, model.targets], feed_dict=feed)
    return recall_scores, map_scores, n_samples.shape[0], top_k, target


def main():
    # 加载数据
    A = load_graph(FLAGS.dataset)
    if FLAGS.use_feats:
        X = load_feats(FLAGS.dataset)
    else:
        X = np.eye(A.shape[0])
    
    num_nodes = A.shape[0]
    if num_nodes % FLAGS.vae_batch_size == 0:
        num_batches_vae = num_nodes // FLAGS.vae_batch_size
    else:
        num_batches_vae = num_nodes // FLAGS.vae_batch_size + 1

    if FLAGS.graph_AE == 'GCN':
        num_batches_vae = 1

    num_nodes = A.shape[0]
    layers_config = list(map(int, FLAGS.vae_layer_config.split(",")))

    train_cascades, train_times = load_cascades(FLAGS.dataset, mode='train')
    val_cascades, val_times = load_cascades(FLAGS.dataset, mode='val')
    test_cascades, test_times = load_cascades(FLAGS.dataset, mode='test')

    train_examples, train_examples_times = get_data_set(train_cascades, train_times, max_len=FLAGS.max_seq_length, mode='train')
    val_examples, val_examples_times = get_data_set(val_cascades, val_times, max_len=FLAGS.max_seq_length, mode='val')
    test_examples, test_examples_times = get_data_set(test_cascades, test_times, max_len=FLAGS.max_seq_length, test_min_percent=FLAGS.test_min_percent, test_max_percent=FLAGS.test_max_percent, mode='test')

    social = InfVAESocial(X.shape[1], A, layers_config, mode='test', feats=X)
    att = InfVAECascades(num_nodes + 1, train_examples, train_examples_times,
                           val_examples, val_examples_times,
                           test_examples, test_examples_times,
                           logging=False, mode='test')
    

    # 创建 Saver 对象
    saver = tf.compat.v1.train.Saver()

    # 恢复模型参数并进行预测
    with tf.compat.v1.Session() as session:

        # input = session.run(att.input)
        # print(input)
        
        session.run(tf.compat.v1.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)      

        # 从文件中恢复模型参数å
        saver.restore(session, "model_checkpoint/android_best61.ckpt")
        print("Models restored from path: model_checkpoint/android_best.ckpt")

         # 准备 z_vae_embeds 的存储空间
        z_vae_embeds = np.zeros([num_nodes + 1, FLAGS.latent_dim])

        sender_embeds =  session.run(att.sender_embeddings)
        receiver_embeds = session.run(att.receiver_embeddings)
        input_feed = social.construct_feed_dict(v_sender_all=sender_embeds,
                                                  v_receiver_all=receiver_embeds, dropout=0.)
        # 获取z_vae_embeds
        for _ in range(0, num_batches_vae):
            vae_embeds, indices = session.run([social.z_mean, social.node_indices], input_feed)
            z_vae_embeds[indices] = vae_embeds
        print("获取潜在变量完成")

        input_feed = att.construct_feed_dict(z_vae_embeddings=z_vae_embeds, is_test=True)

        total_samples = 0
        num_eval_k = len(att.k_list)
        avg_map_scores, avg_recall_scores = [0.] * num_eval_k, [0.] * num_eval_k

        all_outputs, all_targets, outs = [], [],[]

        print("test batchs:",att.num_test_batches)
        for b in range(0, att.num_test_batches):
            # print("b:",b)
            recalls, maps, num_samples, decoder_outputs, decoder_targets = predict(
                session, att, input_feed)
            output = session.run(att.outputs, feed_dict=input_feed)
            # print('output:\n',output)
            all_outputs.append(decoder_outputs)
            outs.append(output)
            all_targets.append(decoder_targets)
            avg_map_scores = list(
                map(operator.add, map(operator.mul, maps,
                                        [num_samples] * num_eval_k), avg_map_scores))
            avg_recall_scores = list(map(operator.add, map(operator.mul, recalls,
                                                            [num_samples] * num_eval_k), avg_recall_scores))
            total_samples += num_samples
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        outs = np.vstack(outs)

        avg_map_scores = list(map(operator.truediv, avg_map_scores, [total_samples] * num_eval_k))
        avg_recall_scores = list(map(operator.truediv, avg_recall_scores, [total_samples] * num_eval_k))

        # print("预测值：%s \n 真实值：%s" % (all_outputs[0],all_targets[0]))
        # print("是不是概率？？",outs[0])

        metrics = dict()
        for k in range(0, num_eval_k):
            K = att.k_list[k]
            metrics["MAP@%d" % K] = avg_map_scores[k]
            metrics["Recall@%d" % K] = avg_recall_scores[k]

        # logger.update_record(avg_map_scores[0], (all_outputs, all_targets, metrics))

        # print evaluation metrics
        # outputs, targets, metrics = logger.best_data
        print("Evaluation metrics on test set:")
        print(metrics)

        # stop queue runners
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
