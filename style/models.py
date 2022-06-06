import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger

#from models import SimpleDecoder, SimpleEncoder, SimpleStyleEncoder
from utils import NUMBER_PERSONS
from losses import standard_style_loss

class ConcatBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConcatBlock, self).__init__()
        self.perceptron = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size)
            )

    def forward(self, x, style):
        if style == None:
            return x
            
        B, D = x.shape
        content_and_style = torch.cat((x, style.repeat(B, 1)), dim=1)
        out = self.perceptron(content_and_style)
        return out + x
    
class CausalDecoder(nn.Module):
    def __init__(self, args, obs_len, fut_len, invariant_dim, style_dim, num_agents, normalize_type='group', decoder_bottle=2) -> None:
        super().__init__()

        self.args = args
        self.obs_len = obs_len
        self.fut_len = fut_len
        self.num_agents = num_agents
        self.invariant_dim = invariant_dim
        self.style_dim = style_dim
        self.normalize_type = normalize_type
        print('normalize_type: ', normalize_type)

        if normalize_type not in ['layer', 'batch' , 'group', 'none']:
            raise ValueError('normalize type: {} not in required list!'.format(normalize_type))
        if normalize_type == 'layer':
            self.inv_norm_layer = nn.LayerNorm(invariant_dim)
            self.style_norm_layer = nn.LayerNorm(style_dim)
        elif normalize_type == 'batch':
            self.inv_norm_layer = nn.BatchNorm1d(invariant_dim)
            self.style_norm_layer = nn.BatchNorm1d(style_dim)
        elif normalize_type == 'group':
            self.inv_norm_layer = nn.GroupNorm(num_groups=num_agents, num_channels=invariant_dim)
            self.style_norm_layer = nn.GroupNorm(num_groups=1, num_channels=style_dim)
        elif normalize_type == 'none':
            self.inv_norm_layer = nn.Identity()
            self.style_norm_layer = nn.Identity()
        
        self.decoder = nn.Sequential(
            nn.Linear(style_dim+invariant_dim, 2*decoder_bottle* style_dim),
            nn.ReLU(),
            nn.Linear(2*decoder_bottle* style_dim, 2*decoder_bottle* style_dim),
            nn.ReLU(),
            nn.Linear(2*decoder_bottle* style_dim, num_agents*2*fut_len)
        )
        
    def forward(self, latent_space, style_feat_space=None):
        # invariant: bz, 2, history_len
        # style:     embedding_dim
        # return: bz, 2*history_len
        latent_space = torch.stack(latent_space.split(2, dim=0), dim=0)
        
        bz = latent_space.size(0)
        latent_space = latent_space.flatten(start_dim=1)
        
        if style_feat_space is not None:
            style_feat_space = style_feat_space.repeat(bz, 1)
        else:
            style_feat_space = torch.zeros(bz, self.style_dim).cuda()
            
        latent_space = self.inv_norm_layer(latent_space)
        style_feat_space = self.style_norm_layer(style_feat_space)
        out = torch.cat((latent_space, style_feat_space), dim=1)
        first_concat = out
        second_concat = None
        
        out = self.decoder(out)
        out = torch.reshape(out, (out.shape[0], self.num_agents, self.fut_len, 2))
        out = out.flatten(start_dim=0, end_dim=1)
        out = torch.permute(out, (1, 0, 2))
        
        
        if self.args.visualize_embedding:
            # Also return two concatenated embeddings
            return out, [first_concat, second_concat]
        return out, [None, None]
              

class GTEncoder(nn.Module):
    def __init__(self, args, style_dim=8):
        super().__init__()
        # style encoder
        self.args = args
        self.style_dim = style_dim
        self.encoder = nn.Sequential(
            nn.Linear(2, args.gt_encoder),
            nn.ReLU(),
            nn.Linear(args.gt_encoder, style_dim)
        )
    def forward(self, x):
        return self.encoder(x)

class SimpleStyleEncoder(nn.Module):
    def __init__(self, args):
        super(SimpleStyleEncoder, self).__init__()

        # style encoder
        self.encoder = nn.Sequential(
            nn.Linear(40, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.args = args
        self.style_dim = 8

        # style classifier
        hidden_size = 8 #50
        feat_space_dim = 8

        # hat classifier above style to get a low dim space for contrastive learning
        self.hat_classifier = nn.Sequential(
            nn.Linear(self.style_dim, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, feat_space_dim)
        )

        # classifier to get error_rate of learned latent space
        self.er_classifier = nn.Sequential(
            nn.Linear(feat_space_dim, feat_space_dim),
            nn.ReLU(), nn.Linear(feat_space_dim, args.classification)
        )
        # associated optimizer. Trained by used train_er_classifier()
        self.er_opt = torch.optim.Adam(self.er_classifier.parameters())


    def train_er_classifier(self,  train_dataset):
        assert (self.args.contrastive)

        for e in tqdm(range(3)):
            train_loaders_iter = [iter(train_loader) for train_loader in train_dataset['loaders']]
            for _ in range(train_dataset['num_batches']):
                batch_loss = []
                self.er_opt.zero_grad() # reset gradients

                for train_loader_iter, loader_name in zip(train_loaders_iter, train_dataset['names']):
                    batch = next(train_loader_iter)
                    with torch.no_grad():
                        low_dim = self.forward(batch[5].cuda(), 'low')
                    class_preds = self.er_classifier(low_dim)
                    label = train_dataset['labels'][loader_name]
                    batch_loss.append(standard_style_loss(class_preds, label))
                
                loss = torch.stack(batch_loss).sum()
                loss.backward()
                self.er_opt.step()


    def forward(self, style_input, what):
        assert(what in set(['low', 'both', 'style', 'class']))
        # for batch size 128
        # style 20 x 128 x 2
        style_input = torch.stack(style_input.split(2, dim=1), dim=1)[:,:,1,:] # 20 x 64 x 2
        style_input = torch.permute(style_input, (1, 0, 2))  # 64 x 20 x 2
        style_input = torch.flatten(style_input, 1) # 64 x 40

        # MLP
        style_seq = self.encoder(style_input)
        # apply reduction without sequences / within batch if needed
        batch_style = style_seq.mean(dim=0).unsqueeze(dim=0)

        if 'style' == what: # only what the style
            return batch_style

        low_seq = self.hat_classifier(style_seq)
        if low_seq.dim()==1: low_seq = torch.nn.functional.normalize(low_seq, dim=0)
        else: low_seq = torch.nn.functional.normalize(low_seq)

        if 'low' == what: # only what the contrastive feat space
            return low_seq
        elif 'both' == what: # both what
            return low_seq, batch_style
        elif 'class' == what:
            class_out = self.er_classifier(low_seq) # only what the class label
            return class_out
        else:
            raise NotImplementedError


class SimpleEncoder(nn.Module):
    def __init__(
            self,
            obs_len,
            hidden_size,
            number_agents,
            bottle_width=4
    ):
        super(SimpleEncoder, self).__init__()

        # num of frames per sequence
        self.obs_len = obs_len

        self.mlp = nn.Sequential(
            nn.Linear(obs_len*number_agents*2, hidden_size*bottle_width),
            nn.ReLU(),
            nn.Linear(hidden_size*bottle_width, hidden_size*bottle_width),
            nn.ReLU(),
            nn.Linear(hidden_size*bottle_width, hidden_size*2),
        )


    def forward(self, obs_traj_rel):

        obs_traj_rel = torch.stack(obs_traj_rel.split(2, dim=1), dim=1)
        obs_traj_rel = torch.permute(obs_traj_rel, (1, 2, 0, 3))
        obs_traj_rel = obs_traj_rel.flatten(start_dim=1)

        encoded = self.mlp(obs_traj_rel)
        
        encoded = torch.stack(encoded.split(encoded.shape[1]//2, dim=1), dim=1)
        encoded = encoded.flatten(start_dim=0, end_dim=1)
        
        return encoded

class ConvEncoder(nn.Module):
    def __init__(
            self,
            obs_len,
            hidden_size,
            number_agents,
    ):
        super(ConvEncoder, self).__init__()

        # num of frames per sequence
        self.obs_len = obs_len

        # input shape: num_agent*bz, 2*obs_len --> 2*bz, hidden_size
        self.encode = nn.Sequential(
            nn.Conv1d(2*obs_len, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, hidden_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(1)
        )
                
    def forward(self, obs_traj_rel):
        # input shape: obs_len, num_agent*bz, 2
        out = torch.einsum('obd->bod', obs_traj_rel)
        out = out.reshape(out.shape[0], -1, 1)
        out = self.encode(out)
        return out
        

class SimpleDecoder(nn.Module):
    def __init__(
            self,
            args,
            obs_len,
            fut_len,
            hidden_size,
            number_of_agents,
            style_input_size=None,
            decoder_bottle=2
    ):
        super(SimpleDecoder, self).__init__()

        # num of frames per sequence
        self.obs_len = obs_len
        self.fut_len = fut_len
        self.args = args
        
        self.style_input_size = style_input_size

        self.noise_fixed = False

        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_size*2, 2*decoder_bottle* hidden_size),
            nn.ReLU(),
            nn.Linear(2*decoder_bottle* hidden_size, 2*decoder_bottle* hidden_size),
            nn.ReLU(),
            nn.Linear(2*decoder_bottle* hidden_size, 2*decoder_bottle* hidden_size),
            nn.ReLU(),
            nn.Linear(2*decoder_bottle* hidden_size, 4* hidden_size)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(4* hidden_size, number_of_agents*decoder_bottle*fut_len),
            nn.ReLU(),
            nn.Linear(number_of_agents*decoder_bottle*fut_len, number_of_agents*decoder_bottle*fut_len),
            nn.ReLU(),
            nn.Linear(number_of_agents*decoder_bottle*fut_len, number_of_agents*decoder_bottle*fut_len),
            nn.ReLU(),
            nn.Linear(number_of_agents*decoder_bottle*fut_len, number_of_agents*2*fut_len)
        )

        self.number_of_agents = number_of_agents

        self.style_blocks = nn.ModuleList(
            [ConcatBlock(self.style_input_size + hidden_size*2, hidden_size*2),
            ConcatBlock(self.style_input_size + 4* hidden_size, 4* hidden_size)]
        )

    def forward(self, latent_space, style_feat_space=None):

        traj_lstm_hidden_state = torch.stack(latent_space.split(2, dim=0), dim=0)
        out = traj_lstm_hidden_state.flatten(start_dim=1)

        if style_feat_space != None:
            out = self.style_blocks[0](out, style_feat_space)
        first_concat = out
        out = self.mlp1(out)

        if style_feat_space != None:
            out = self.style_blocks[1](out, style_feat_space)
        second_concat = out
        
        out = self.mlp2(out)

        out = torch.reshape(out, (out.shape[0], self.number_of_agents, self.fut_len, 2))

        out = out.flatten(start_dim=0, end_dim=1)

        out = torch.permute(out, (1, 0, 2))

        if self.args.visualize_embedding:
            # Also return two concatenated embeddings
            return out, [first_concat, second_concat]
        return out, [None, None]


class CausalMotionModel(nn.Module):
    def __init__(self, args):
        super(CausalMotionModel, self).__init__()

        latent_space_size = 8
        self.args = args
        self.inv_encoder = SimpleEncoder(args.obs_len, latent_space_size, NUMBER_PERSONS)
        self.style_encoder = SimpleStyleEncoder(args)
        self.gt_encoder = GTEncoder(args)
        if args.causal_decoder:        
            self.decoder = CausalDecoder(
                args,
                args.obs_len,
                args.fut_len,
                latent_space_size*2,
                style_dim=self.gt_encoder.style_dim if args.gt_style else self.style_encoder.style_dim,
                num_agents=NUMBER_PERSONS,
                normalize_type=args.norm_type,
                decoder_bottle=args.decoder_bottle
            )
        else:
            self.decoder = SimpleDecoder(
                args,
                args.obs_len,
                args.fut_len,
                latent_space_size,
                NUMBER_PERSONS,
                style_input_size=self.gt_encoder.style_dim if args.gt_style else self.style_encoder.style_dim,
                decoder_bottle=args.decoder_bottle
            )
        self.visualize_embedding = args.visualize_embedding

    def forward(self, batch, training_step, gt_style=None, inspect=False):
        assert (training_step in ['P3', 'P4', 'P5', 'P6'])

        (obj_traj, _, _, _, _, style_input, _) = batch
        if gt_style == None and self.args.gt_style and training_step=='P4':
            raise ValueError('The gt style is not provided')
        # compute only style and classify
        if training_step == 'P4':
            if self.training:
                if self.args.gt_style:
                    style_embedding = self.gt_encoder(gt_style)
                    latent_content_space = self.inv_encoder(obj_traj)
                    output, [first_concat, second_concat] = self.decoder(latent_content_space, style_feat_space=style_embedding)
                    if inspect:
                        return output, [latent_content_space, first_concat, second_concat]
                    return output
                else:
                    return self.style_encoder(style_input, 'low')
            else:
                if self.args.gt_style:
                    style_embedding = self.gt_encoder(gt_style)
                    latent_content_space = self.inv_encoder(obj_traj)
                    output, [first_concat, second_concat] = self.decoder(latent_content_space, style_feat_space=style_embedding)
                    if inspect:
                        return output, [latent_content_space, first_concat, second_concat] 
                    return output
                else:
                    return self.style_encoder(style_input, 'class')

        # compute invariants
        latent_content_space = self.inv_encoder(obj_traj)

        # compute style if required
        style_encoding = None
        if training_step in ['P5', 'P6']:
            low_dim, style_encoding = self.style_encoder(style_input, 'both')

        # compute prediction
        output, [first_concat, second_concat] = self.decoder(latent_content_space, style_feat_space=style_encoding)

        if training_step == 'P6' and (self.training or self.visualize_embedding):
            if inspect:
                return output, low_dim, [latent_content_space, first_concat, second_concat]  # need the low_dim to keep training contrastive loss
            return output, low_dim
        else:
            if inspect:
                return output, [latent_content_space, first_concat, second_concat]
            return output
