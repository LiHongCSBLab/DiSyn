import os
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
    log_loss
from torch import nn, autograd
from torch.utils.data import TensorDataset, DataLoader

from net import EncoderDecoder, FC
from utils import auprc, repeat2d_to, repeat_to, generate_dsn, dsn_loss, feature_concat


def data_synthesize(s_disynae, t_disynae, train_dataloader, test_dataloader, di_scale_co, device):
    # ################################################## Di synthesis ##################################################
    # Here we take train_dataloader (4/5 of GDSC labeld data) and unlabeled_test_dataloader (TCGA data) to synthesize new labeled Syn_data
    # Then merge it into the original train_dataset to regenerate a train_loader
    syn_x = []
    syn_y = []

    for i in range(di_scale_co):
        for step, s_batch in enumerate(train_dataloader):
            # training model using target data
            t_batch = next(iter(test_dataloader))

            s_x, s_y = s_batch[0].to(device), s_batch[1].to(device)
            t_x = t_batch[0].to(device)

            # If the lengths of two batches are different, the alignment will be adapted first.
            if len(s_x) > len(t_x):
                t_x = repeat2d_to(t_x, len(s_x))
            elif len(s_x) < len(t_x):
                s_x = repeat2d_to(s_x, len(t_x))
                s_y = repeat_to(s_y, len(t_x))

            source_shared_codes = s_disynae.s_encode(s_x)
            target_private_codes = t_disynae.p_encode(t_x)
            concat_feature = torch.cat((target_private_codes, source_shared_codes), dim=1)
            synsample = t_disynae.decode(concat_feature).detach()

            syn_x.append(synsample)
            syn_y.append(s_y)

    syn_x = torch.cat(syn_x, dim=0).detach().cpu()
    syn_y = torch.cat(syn_y, dim=0).detach().cpu()

    enhanced_X = torch.cat([train_dataloader.dataset.tensors[0], syn_x], dim=0)
    enhanced_Y = torch.cat([train_dataloader.dataset.tensors[1], syn_y], dim=0)

    train_labeled_gdsc_dateset = TensorDataset(enhanced_X, enhanced_Y)
    train_dataloader = DataLoader(train_labeled_gdsc_dateset, batch_size=train_dataloader.batch_size, shuffle=True)


    return train_dataloader


def dsn_recon_step(dsn_source, dsn_target, s_batch, t_batch, device, optimizer, scheduler=None):
    dsn_source.zero_grad()
    dsn_target.zero_grad()
    dsn_source.train()
    dsn_target.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss = dsn_loss(*dsn_source(s_x))
    t_loss = dsn_loss(*dsn_target(t_x))

    optimizer.zero_grad()

    loss = s_loss + t_loss
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return


def save_check(history, reset_count=1):
    # Method from https://github.com/XieResearchGroup/CODE-AE/blob/6dc17a5f3b7ce2e89736d1d575fb75951bd2c9ea/code/fine_tuning.py.
    save_flag = False
    stop_flag = False

    metric_name = 'auroc'

    if 'best_index' not in history:
        history['best_index'] = 0
    if metric_name.endswith('loss'):
        if history[metric_name][-1] < history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
    else:
        if history[metric_name][-1] > history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1

    if len(history[metric_name]) - history['best_index'] > 5 * reset_count and history['best_index'] > 0:
        stop_flag = True

    return save_flag, stop_flag


def evaluate_target_classification_epoch(classifier, dataloader, device, history):
    y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
            y_pred = torch.sigmoid(classifier(x_batch)).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds))
    history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
    history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['bce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))

    return history


def dsn_dann_train_step(critic, dsn_source, dsn_target, s_batch, t_batch, device, optimizer, scheduler=None):
    critic.zero_grad()
    dsn_source.zero_grad()
    dsn_target.zero_grad()
    critic.eval()
    dsn_source.train()
    dsn_target.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    t_code = dsn_target.s_encode(t_x)

    optimizer.zero_grad()

    critic_loss = -torch.mean(critic(t_code))

    s_loss = dsn_loss(*dsn_source(s_x))
    t_loss = dsn_loss(*dsn_target(t_x))

    recons_loss = s_loss + t_loss

    loss = recons_loss + critic_loss

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return


def disyn_train_step_backward(critic, s_disynae, t_disynae, s_batch, t_batch, device, optimizer, alpha, loss_function, scheduler=None):
    critic.zero_grad()

    s_disynae.zero_grad()
    t_disynae.zero_grad()

    critic.eval()

    s_disynae.train()
    t_disynae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)
    t_y = t_batch[1].to(device)

    t_code_p = t_disynae.p_encode(t_x)

    optimizer.zero_grad()

    t_y = torch.tensor([int(x) for x in t_y])
    label_one_hot = torch.nn.functional.one_hot(t_y, 2).float().to(device)

    class_adv_loss = - loss_function(critic(t_code_p), label_one_hot)

    s_loss = dsn_loss(*s_disynae(s_x))
    t_loss = dsn_loss(*t_disynae(t_x))
    recons_loss = s_loss + t_loss

    loss = recons_loss + alpha * class_adv_loss
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return


def classification_train_step(model, batch, loss_fn, device, optimizer, scheduler=None):
    model.zero_grad()
    model.train()

    x = batch[0].to(device)
    y = batch[1].to(device)

    loss = loss_fn(model(x).squeeze().float(), y.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return


def critic_dsn_train_step(critic, dsn_source, dsn_target, s_batch, t_batch, device, optimizer, scheduler=None):
    critic.zero_grad()
    dsn_source.zero_grad()
    dsn_target.zero_grad()
    dsn_source.eval()
    dsn_target.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = dsn_source.s_encode(s_x)
    t_code = dsn_target.s_encode(t_x)

    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return


def class_adv_train_step(critic, s_disynae, t_disynae, s_batch, t_batch, device, optimizer, loss_function, scheduler=None):
    # Train the class confounder in di recon training
    # This process must be done separately for each drug
    # Although it is in the unsupervised pre-training code, it is already in a supervised loop and should be left alone. It will be left here for now.
    # Because this must be done separately for each drug, the GDSC used for synthesizing comes from each drug individually.

    critic.zero_grad()
    s_disynae.zero_grad()
    t_disynae.zero_grad()
    s_disynae.eval()
    t_disynae.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    s_y = s_batch[1].to(device)
    t_x = t_batch[0].to(device)
    t_y = t_batch[1].to(device)

    s_code = s_disynae.p_encode(s_x)
    t_code = t_disynae.p_encode(t_x)

    t_y = torch.tensor([int(x) for x in t_y])
    label_one_hot = torch.nn.functional.one_hot(t_y, 2).float().to(device)
    
    loss = loss_function(critic(t_code), label_one_hot)  # + loss_function(critic(s_code), s_y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return


def pretrain_proc_disyn_adv(s_dataloader, t_dataloader, **kwargs):
    """
    The first step of unsupervised pre-training in Disyn
    """
    source_dsn, target_dsn = generate_dsn(**kwargs)

    da_discriminator = FC(input_dim=128,
                          output_dim=1,
                          hidden_dims=[64, 32],
                          drop_out=kwargs['drop_out']).to(kwargs['device'])

    ae_params = [target_dsn.private_encoder.parameters(),
                 source_dsn.private_encoder.parameters(),
                 source_dsn.decoder.parameters(),
                 source_dsn.shared_encoder.parameters()
                 ]
    t_ae_params = [target_dsn.private_encoder.parameters(),
                   source_dsn.private_encoder.parameters(),
                   source_dsn.decoder.parameters(),
                   source_dsn.shared_encoder.parameters()
                   ]

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=0.0001)
    discriminator_optimizer = torch.optim.RMSprop(da_discriminator.parameters(), lr=0.0001)
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=0.0001)

    for epoch in range(int(kwargs['nums_recon'])):
        for step, s_batch in enumerate(s_dataloader):
            t_batch = next(iter(t_dataloader))

            dsn_recon_step(dsn_source=source_dsn, dsn_target=target_dsn, s_batch=s_batch, t_batch=t_batch, device=kwargs['device'], optimizer=ae_optimizer)

        if epoch % 50 == 0:
            print(f'recon epochs {epoch}')

    for epoch in range(int(kwargs['nums_critic'])):
        for step, s_batch in enumerate(s_dataloader):
            t_batch = next(iter(t_dataloader))
            critic_dsn_train_step(critic=da_discriminator, dsn_source=source_dsn, dsn_target=target_dsn, s_batch=s_batch, t_batch=t_batch, device=kwargs['device'],
                                  optimizer=discriminator_optimizer)

            if (step + 1) % 5 == 0:
                dsn_dann_train_step(critic=da_discriminator, dsn_source=source_dsn, dsn_target=target_dsn, s_batch=s_batch, t_batch=t_batch, device=kwargs['device'],
                                    optimizer=t_ae_optimizer)
        if epoch % 50 == 0:
            print(f'critic train {epoch}')

    torch.save(source_dsn.state_dict(), os.path.join(kwargs['model_save_path'], 'source_dsn.pt'))
    torch.save(target_dsn.state_dict(), os.path.join(kwargs['model_save_path'], 'target_dsn.pt'))

    return target_dsn.shared_encoder


def task_specific_train_di_adv(source_dsn, source_labeled_train_dataloader, source_labeled_val_dataloader, target_test_dataloader, fold_count, metrics_dict, **kwargs):
    """
    task_specific_train function
    """
    target_decoder = FC(input_dim=128, output_dim=1, hidden_dims=[64, 32]).to(kwargs['device'])
    target_classifier = EncoderDecoder(encoder=source_dsn.shared_encoder, decoder=target_decoder).to(kwargs['device'])

    val_history = defaultdict(list)
    test_history = defaultdict(list)

    encoder_module_indices = [i for i in range(len(list(source_dsn.shared_encoder.modules())))
                              if str(list(source_dsn.shared_encoder.modules())[i]).startswith('Linear')]

    classification_loss = nn.BCEWithLogitsLoss()
    target_classification_params = [target_classifier.decoder.parameters()]
    target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=kwargs['ft_lr'])

    torch.save(target_classifier.state_dict(), os.path.join(kwargs['task_save_path'], f'classifier_{fold_count}.pt'))

    lr = kwargs['ft_lr']
    reset_count = 1
    for epoch in range(2000):
        if epoch % 50 == 0:
            print(f'classifier train epoch {epoch}')

        for step, s_batch in enumerate(source_labeled_train_dataloader):
            classification_train_step(model=target_classifier, 
                                      batch=s_batch, 
                                      loss_fn=classification_loss, 
                                      device=kwargs['device'],
                                      optimizer=target_classification_optimizer)

        val_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                           dataloader=source_labeled_val_dataloader,
                                                           device=kwargs['device'],
                                                           history=val_history)

        test_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                            dataloader=target_test_dataloader,
                                                            device=kwargs['device'],
                                                            history=test_history)

        save_flag, stop_flag = save_check(history=val_history, reset_count=reset_count)

        if save_flag:
            torch.save(target_classifier.state_dict(), os.path.join(kwargs['task_save_path'], f'classifier_{fold_count}.pt'))

        if stop_flag:
            try:
                ind = encoder_module_indices.pop()
                target_classifier.load_state_dict(
                    torch.load(os.path.join(kwargs['task_save_path'], f'classifier_{fold_count}.pt'), map_location=torch.device(kwargs['device'])))
                target_classification_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
                reset_count += 1

            except:
                break

    metrics_dict['best_index'].append(val_history['best_index'])

    for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
        metrics_dict[metric].append(test_history[metric][val_history['best_index']])

    return metrics_dict


def recon_classadv_train_di_adv(dsn_source, dsn_target, s_labeled_dataloader, t_unlabeled_dataloader, fold_count, **kwargs):
    class_confounding_classifier = FC(input_dim=128,
                                      output_dim=2,
                                      hidden_dims=[64, 32],
                                      drop_out=kwargs['drop_out']).to(kwargs['device'])

    ae_params = [dsn_target.private_encoder.parameters(),
                 dsn_source.private_encoder.parameters(),
                 dsn_target.decoder.parameters()]

    t_ae_params = [dsn_target.private_encoder.parameters(),
                   dsn_source.private_encoder.parameters(),
                   dsn_target.decoder.parameters()]

    class_classifier_loss = torch.nn.CrossEntropyLoss()

    class_classifier_optimizer = torch.optim.RMSprop(class_confounding_classifier.parameters(), lr=0.0001)

    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=0.0001)
    t_ae_optimizer = torch.optim.RMSprop(chain(*t_ae_params), lr=0.0001)

    for epoch in range(kwargs['clf_train_bf_recon']):
        for step, s_batch in enumerate(s_labeled_dataloader):
            t_batch = next(iter(t_unlabeled_dataloader))
            t_batch = feature_concat(dsn_source, dsn_target, s_batch, t_batch, kwargs['device'])

            class_adv_train_step(critic=class_confounding_classifier, s_disynae=dsn_source, t_disynae=dsn_target, s_batch=s_batch, t_batch=t_batch, device=kwargs['device'],
                                 optimizer=class_classifier_optimizer, loss_function=class_classifier_loss)

    for epoch in range(kwargs['recon_epochs']):
        if epoch % 50 == 0:
            print(f'recon epoch {epoch}')
        for step, s_batch in enumerate(s_labeled_dataloader):

            t_batch = next(iter(t_unlabeled_dataloader))
            t_batch = feature_concat(dsn_source, dsn_target, s_batch, t_batch, kwargs['device'])

            dsn_recon_step(dsn_source=dsn_source, dsn_target=dsn_target, s_batch=s_batch, t_batch=t_batch, device=kwargs['device'], optimizer=ae_optimizer)

            class_adv_train_step(critic=class_confounding_classifier,
                                 s_disynae=dsn_source,
                                 t_disynae=dsn_target,
                                 s_batch=s_batch,
                                 t_batch=t_batch,
                                 device=kwargs['device'],
                                 optimizer=class_classifier_optimizer,
                                 loss_function=class_classifier_loss)

            if (step + 1) % kwargs['rev_back_internal'] == 0:
                disyn_train_step_backward(critic=class_confounding_classifier,
                                          s_disynae=dsn_source,
                                          t_disynae=dsn_target,
                                          s_batch=s_batch,
                                          t_batch=t_batch,
                                          device=kwargs['device'],
                                          optimizer=t_ae_optimizer,
                                          alpha=kwargs['clsadv_alpha'],
                                          loss_function=class_classifier_loss)

    os.makedirs(os.path.join(kwargs['task_save_path']), exist_ok=True)
    torch.save(dsn_source.state_dict(), os.path.join(kwargs['task_save_path'], f's_disyn_adv_recon_{fold_count}.pt'))
    torch.save(dsn_target.state_dict(), os.path.join(kwargs['task_save_path'], f't_disyn_adv_recon_{fold_count}.pt'))

    return


def task_specific_train_disyn_step2(dsn_source, dsn_target, s_labeled_train_dataloader, s_labeled_val_dataloader, t_unlabeled_dataloader,
                                    t_test_dataloader, fold_count, metrics_dict, **kwargs):
    target_decoder = FC(input_dim=kwargs['latent_dim_task'], output_dim=1, hidden_dims=kwargs['linker_dim']).to(kwargs['device'])
    target_decoder.module = dsn_source.adapter
    target_classifier = EncoderDecoder(encoder=dsn_source.shared_encoder, decoder=target_decoder).to(kwargs['device'])

    val_history = defaultdict(list)
    test_history = defaultdict(list)

    encoder_module_indices = [i for i in range(len(list(dsn_source.shared_encoder.modules())))
                              if str(list(dsn_source.shared_encoder.modules())[i]).startswith('Linear')]

    lr = kwargs['ft_lr']

    classification_loss = nn.BCEWithLogitsLoss()
    target_classification_params = [target_classifier.decoder.parameters()]
    target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)

    ae_params = [dsn_target.private_encoder.parameters(),
                 dsn_source.private_encoder.parameters(),
                 dsn_source.decoder.parameters()]
    ae_optimizer = torch.optim.AdamW(chain(*ae_params), lr=kwargs['ft_lr'])

    if kwargs['di_scale_co']:
        print('Train_dataloader data size before reconstruction', len(s_labeled_train_dataloader.dataset.tensors[1]))
        # The premise here is that train set has more samples than target set
        print('di_scale_co', kwargs['di_scale_co'])

        synthe_dataloader = data_synthesize(dsn_source, dsn_target, s_labeled_train_dataloader, t_unlabeled_dataloader, kwargs['di_scale_co'], kwargs['device'])
        synthe_val_dataloader = data_synthesize(dsn_source, dsn_target, s_labeled_val_dataloader, t_unlabeled_dataloader, kwargs['di_scale_co'], kwargs['device'])

        print('Train_dataloader data size after reconstruction', len(synthe_dataloader.dataset.tensors[1]))
    else:
        synthe_dataloader = s_labeled_train_dataloader
        synthe_val_dataloader = s_labeled_val_dataloader

    torch.save(target_classifier.state_dict(), os.path.join(kwargs['task_save_path'], f'classifier_{fold_count}.pt'))

    reset_count = 1
    for epoch in range(2000):
        if epoch % 50 == 0:
            print(f'train epoch {epoch}')
        for step, synthe_batch in enumerate(synthe_dataloader):
            classification_train_step(model=target_classifier,
                                      batch=synthe_batch,
                                      loss_fn=classification_loss,
                                      device=kwargs['device'],
                                      optimizer=target_classification_optimizer)

        # if (step + 1) % 3 == 0:
        #     recon_s_batch = next(iter(s_unlabeled_dataloader))
        #     recon_t_batch = next(iter(t_unlabeled_dataloader))
        #
        #     dsn_recon_step(s_disynae=s_disynae,
        #                    t_disynae=t_disynae,
        #                    s_batch=recon_s_batch,
        #                    t_batch=recon_t_batch,
        #                    device=kwargs['device'],
        #                    optimizer=ae_optimizer)

        val_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                           dataloader=synthe_val_dataloader,
                                                           device=kwargs['device'],
                                                           history=val_history)

        test_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                            dataloader=t_test_dataloader,
                                                            device=kwargs['device'],
                                                            history=test_history)

        save_flag, stop_flag = save_check(history=val_history, reset_count=reset_count)

        if save_flag:
            torch.save(target_classifier.state_dict(), os.path.join(kwargs['task_save_path'], f'classifier_{fold_count}.pt'))

        if stop_flag:
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing {epoch}')
                target_classifier.load_state_dict(torch.load(os.path.join(kwargs['task_save_path'], f'classifier_{fold_count}.pt'), map_location=torch.device(kwargs['device'])))
                target_classification_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
                reset_count += 1
            except:
                break

    metrics_dict['best_index'].append(val_history['best_index'])
    # The test set result of the best point on val_set is recorded
    for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
        metrics_dict[metric].append(test_history[metric][val_history['best_index']])

    return metrics_dict
