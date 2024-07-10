import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve
import torch.nn.functional as F

from DiSyn import DiSyn
from net import DSN, EncoderDecoder, FC
import path_config


def repeat2d_to(tensor, length):
    multiple, remains = divmod(length, len(tensor))
    return torch.cat([tensor.repeat(multiple, 1), tensor[:remains]], dim=0)


def repeat_to(tensor, length):
    multiple, remains = divmod(length, len(tensor))
    return torch.cat([tensor.repeat(multiple), tensor[:remains]], dim=0)


def set_to_str(args_dict, param_set):
    return "_".join(["_".join([k, str(args_dict[k])]) for k in param_set])


def feature_concat(s_disynae, t_disynae, s_batch, t_batch, device):
    # Synthesize data in batch
    source_shared_codes = s_disynae.s_encode(s_batch[0].to(device))
    target_private_codes = t_disynae.p_encode(t_batch[0].to(device))
    s_y = s_batch[1]

    if len(source_shared_codes) > len(target_private_codes):
        target_private_codes = repeat2d_to(target_private_codes, len(source_shared_codes))
    elif len(source_shared_codes) < len(target_private_codes):
        source_shared_codes = repeat2d_to(source_shared_codes, len(target_private_codes))
        s_y = repeat_to(s_batch[1], len(target_private_codes))

    concat_feature = torch.cat((target_private_codes, source_shared_codes), dim=1)

    return t_disynae.decoder(concat_feature).detach(), s_y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def check_postfix(Dir, postfix):
    Files = os.listdir(Dir)
    for k in range(len(Files)):
        Files[k] = os.path.splitext(Files[k])[-1]
    return True if postfix in Files else False


def load_classifier(model_save_path, fold_count, **kwargs):
    shared_encoder = FC(input_dim=kwargs['input_dim'],
                        output_dim=128,
                        hidden_dims=[512, 256],
                        drop_out=kwargs['drop_out']).to(kwargs['device'])

    decoder = FC(input_dim=128,
                 output_dim=1,
                 hidden_dims=[64, 32],
                 drop_out=kwargs['drop_out']).to(kwargs['device'])

    target_classifier = EncoderDecoder(encoder=shared_encoder, decoder=decoder).to(kwargs['device'])

    target_classifier.load_state_dict(torch.load(os.path.join(model_save_path, f'classifier_{fold_count}.pt'), map_location=torch.device(kwargs['device'])))

    return target_classifier


def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)


def dsn_loss(input, recons, z):
    # dsn_loss from
    # https://github.com/fungtion/DSN
    # https://github.com/XieResearchGroup/CODE-AE
    p_z = z[:, :z.shape[1] // 2]
    s_z = z[:, z.shape[1] // 2:]

    recons_loss = F.mse_loss(input, recons)

    s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
    s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

    p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
    p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

    ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))

    loss = recons_loss + ortho_loss
    return loss


def generate_dsn(**kwargs):
    shared_encoder = FC(input_dim=kwargs['input_dim'],
                        output_dim=128,
                        hidden_dims=[512, 256],
                        drop_out=kwargs['drop_out']).to(kwargs['device'])

    shared_decoder = FC(input_dim=2 * 128,
                        output_dim=kwargs['input_dim'],
                        hidden_dims=[256, 512],
                        drop_out=kwargs['drop_out']).to(kwargs['device'])

    source_dsn = DSN(shared_encoder=shared_encoder,
                     decoder=shared_decoder,
                     input_dim=kwargs['input_dim'],
                     latent_dim=128,
                     hidden_dims=[512, 256],
                     drop_out=kwargs['drop_out']).to(kwargs['device'])

    target_dsn = DSN(shared_encoder=shared_encoder,
                     decoder=shared_decoder,
                     input_dim=kwargs['input_dim'],
                     latent_dim=128,
                     hidden_dims=[512, 256],
                     drop_out=kwargs['drop_out']).to(kwargs['device'])

    return source_dsn, target_dsn


def load_dsn(**kwargs):
    source_dsn, target_dsn = generate_dsn(**kwargs)

    source_dsn.load_state_dict(torch.load(os.path.join(kwargs['model_path'], f'source_dsn.pt'), map_location=torch.device(kwargs['device'])))
    target_dsn.load_state_dict(torch.load(os.path.join(kwargs['model_path'], f'target_dsn.pt'), map_location=torch.device(kwargs['device'])))

    assert (id(source_dsn.shared_encoder) == id(target_dsn.shared_encoder))
    assert (id(source_dsn.decoder) == id(target_dsn.decoder))

    return source_dsn, target_dsn


def generate_disyn(**kwargs):
    shared_encoder = FC(input_dim=kwargs['input_dim'],
                        output_dim=128,
                        hidden_dims=[512, 256],
                        drop_out=kwargs['drop_out']).to(kwargs['device'])

    shared_decoder = FC(input_dim=128 + kwargs['latent_dim_task'],
                        output_dim=kwargs['input_dim'],
                        hidden_dims=[256, 512],
                        drop_out=kwargs['drop_out']).to(kwargs['device'])

    shared_encoder_linker = FC(input_dim=128,
                               output_dim=kwargs['latent_dim_task'],
                               hidden_dims=kwargs['linker_dim'],
                               drop_out=kwargs['drop_out']).to(kwargs['device']).module

    s_disynae = DiSyn(shared_encoder=shared_encoder,
                    adapter=shared_encoder_linker,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=128,
                    hidden_dims=[512, 256],
                    drop_out=kwargs['drop_out']).to(kwargs['device'])

    t_disynae = DiSyn(shared_encoder=shared_encoder,
                    adapter=shared_encoder_linker,
                    decoder=shared_decoder,
                    input_dim=kwargs['input_dim'],
                    latent_dim=128,
                    hidden_dims=[512, 256],
                    drop_out=kwargs['drop_out']).to(kwargs['device'])

    return s_disynae, t_disynae


def build_disyn_from_classifier(model_save_path, fold_count, **kwargs):
    s_disynae, t_disynae = generate_disyn(**kwargs)

    target_classifier = load_classifier(model_save_path, fold_count, **kwargs)

    s_disynae.shared_encoder.load_state_dict(target_classifier.encoder.state_dict())
    t_disynae.shared_encoder.load_state_dict(target_classifier.encoder.state_dict())
    s_disynae.adapter.load_state_dict(target_classifier.decoder.module.state_dict())
    t_disynae.adapter.load_state_dict(target_classifier.decoder.module.state_dict())

    assert (id(s_disynae.shared_encoder) == id(t_disynae.shared_encoder))
    assert (id(s_disynae.adapter) == id(t_disynae.adapter))

    return s_disynae, t_disynae


def load_disyn(fold_count, **kwargs):
    s_disyn, t_disyn = generate_disyn(**kwargs)

    s_disyn.load_state_dict(torch.load(os.path.join(kwargs['model_path'], f's_disyn_adv_recon_{fold_count}.pt'), map_location=torch.device(kwargs['device'])))
    t_disyn.load_state_dict(torch.load(os.path.join(kwargs['model_path'], f't_disyn_adv_recon_{fold_count}.pt'), map_location=torch.device(kwargs['device'])))

    assert (id(s_disyn.shared_encoder) == id(t_disyn.shared_encoder))
    assert (id(s_disyn.decoder) == id(t_disyn.decoder))

    return s_disyn, t_disyn


def load_classifier_for_inference(path, **kwargs):
    shared_encoder = FC(input_dim=kwargs['input_dim'],
                        output_dim=128,
                        hidden_dims=[512, 256]).to(kwargs['device'])

    decoder = FC(input_dim=128,
                 output_dim=1,
                 hidden_dims=[64, 32]).to(kwargs['device'])

    drug_mapping_df = pd.read_csv(path_config.gdsctcga_mapping_file, index_col=0)

    drug = drug_mapping_df.loc[kwargs['drug'], 'drug_name']
    model_path = os.path.join(path, str(drug)+'.pt')

    target_classifier = EncoderDecoder(encoder=shared_encoder, decoder=decoder).to(kwargs['device'])

    target_classifier.load_state_dict(torch.load(model_path, map_location=torch.device(kwargs['device'])))

    return target_classifier


def predict_target_classification(classifier, test_df, device):
    y_preds = np.array([])
    classifier.eval()

    for df in [test_df[i:i+64] for i in range(0,test_df.shape[0],64)]:
        x_batch = torch.from_numpy(df.values.astype('float32')).to(device)
        with torch.no_grad():
            y_pred = torch.sigmoid(classifier(x_batch)).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    output_df = pd.DataFrame(y_preds,index=test_df.index,columns=['score'])
    
    return output_df


def get_model_save_path(args):
    if args.step == 1:
        return os.path.join(path_config.result_path, f'Disyn_{args.seed}', f'task_specific_train_step{args.step - 1}', args.drug, f'ft_lr_{args.ft_lr}')
    else:
        params_list = ['di_scale_co', 'drop_out', 'ft_lr']
        task_param_str = set_to_str(vars(args), params_list)
        return os.path.join(path_config.result_path, f'Disyn_{args.seed}', f'task_specific_train_step{args.step - 1}', args.drug, task_param_str)