import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


from .utils import CATransformerBlockTest
import math

def point_dispersion_loss(coords, delta=0.1, alpha=30.0,
                          lambda_rep=1.0, lambda_disk=1.0,
                          lambda_edge=0.5, lambda_mean=0.01,
                          eps=1e-6):
    """
    coords: (B, Q, N, 2) in [-1, 1]
    returns: scalar loss
    """
    B, Q, N, _ = coords.shape
    device = coords.device

    # -----------------------------
    # Pairwise distances (B,Q,N,N)
    # -----------------------------
    diffs = coords.unsqueeze(3) - coords.unsqueeze(2)   # (B,Q,N,N,2)
    D2 = (diffs ** 2).sum(-1)                           # (B,Q,N,N)
    D = torch.sqrt(D2 + eps)

    # mask to remove self-distances
    mask = ~torch.eye(N, dtype=torch.bool, device=device)  # (N,N)
    mask = mask.unsqueeze(0).unsqueeze(0)                  # (1,1,N,N)

    # -----------------------------
    # 1. Gaussian repulsion
    # -----------------------------
    L = 2 * (1 - delta)
    A = L * L
    tau = 0.8 * torch.sqrt(torch.tensor(A / N, device=device))
    rep = torch.exp(-D2 / (tau ** 2))
    L_rep = (rep * mask).sum() / (B * Q * N * (N - 1))

    # -----------------------------
    # 2. Poisson-disk style min distance penalty
    # -----------------------------
    # r_bar = torch.sqrt(torch.tensor(A / N, device=device))
    # r_min = 0.85 * r_bar
    # disk_pen = F.relu(r_min - D) ** 2
    # L_disk = (disk_pen * mask).sum() / (B * Q * N * (N - 1))

    # -----------------------------
    # 3. Edge soft barrier
    # -----------------------------
    d_edge = torch.minimum(1 - coords[..., 0].abs(),
                           1 - coords[..., 1].abs())     # (B,Q,N)
    L_edge = F.softplus(alpha * (delta - d_edge)).mean()

    # -----------------------------
    # 4. Mean regularizer (prevent drift)
    # -----------------------------
    # mean = coords.mean(dim=2)                            # (B,Q,2)
    # L_mean = (mean ** 2).sum(-1).mean()

    # -----------------------------
    # Total
    # -----------------------------
    loss = (lambda_rep * L_rep +
            # lambda_disk * L_disk +
            lambda_edge * L_edge) #+
            # lambda_mean * L_mean)
    return loss

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_resid(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,bias=True):
        super().__init__()
        assert in_features == hidden_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features,bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x_in):
        x = self.fc1(x_in)
        x = self.act(x)
        x = self.drop(x)+x_in
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Conv_resid(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,bias=True):
        super().__init__()
        assert in_features == hidden_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features,3,1,1,bias=bias)
        #self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, out_features,3,1,1,bias=bias)
        #self.drop = nn.Dropout(drop)

    def forward(self, x_in):
        x = self.conv1(x_in)+x_in
        #x = self.act(x)
        x = self.conv2(x)
        #x = self.fc2(x)
        #x = self.drop(x)
        return x
    
class MyModel(nn.Module):
    def __init__(self, backbone,text_encoder,sample_expression_num,layer_dim_list=None,text_dim=768,text_len=25,frame_num=4,last_patch_h=2,qnum=1,cfg=None):
        super().__init__()
        self.text_dim=text_dim
        self.text_len=text_len #12 25
        self.frame_num=frame_num
        self.last_patch_h = last_patch_h
        self.qnum=qnum
        
        self.text_encoder = text_encoder
        
        self.backbone = backbone
        if layer_dim_list is None:
            self.num_layers = len(backbone.layers)
            self.layer_dim_list = [backbone.layers[i].dim for i in range(self.num_layers)]
            
        else:
            self.layer_dim_list = layer_dim_list
            self.num_layers = len(self.layer_dim_list)
        
        # if self.backbone.head is not None:
        #     self.backbone.head=None
        # if self.backbone.norm is not None:
        #     self.backbone.norm=None
            

        self.output_proj_ped=nn.ModuleList()
        # self.output_proj_ped_avg=nn.ModuleList()

        self.objf_resid_down=nn.ModuleList()
        self.output_ca_qkv=nn.ModuleList()
        self.pos_avg_pool=nn.ModuleList()
        self.output_obj_norm_ped = nn.ModuleList()
        self.head=nn.ModuleList()
        
        self.absolute_pos_objf_embed=nn.ParameterList()

        self.condition_num = 10
        self.conditional_pos_embed = nn.Parameter(torch.zeros(1, self.condition_num, self.text_dim))
        nn.init.normal_(self.conditional_pos_embed, std=.02)
        self.conditional_pos_decoder = nn.ModuleList()
        self.conditional_pos_projector = nn.ModuleList()
        self.conditional_f_proj = nn.ModuleList()
        self.conditional_f_norm = nn.ModuleList()

        self.ptHlist = []
        for ip in list(range(self.num_layers))[::-1]:
            self.ptHlist.append(self.last_patch_h*2**(ip))
        
        for i_layer in range(self.num_layers):
            self.output_ca_qkv.append(CATransformerBlockTest(layer_id=i_layer,dim=self.text_dim,n_heads=8,norm_eps=None,drop_out=0.0))
            self.conditional_pos_decoder.append(CATransformerBlockTest(layer_id=i_layer,dim=self.text_dim,n_heads=8,norm_eps=None,drop_out=0.0))
            self.conditional_pos_projector.append(nn.Linear(self.text_dim, 2))

            self.conditional_f_proj.append(Mlp_resid(self.layer_dim_list[i_layer],self.layer_dim_list[i_layer],self.text_dim))
            self.conditional_f_norm.append(nn.LayerNorm(self.layer_dim_list[i_layer]))

            self.output_obj_norm_ped.append(nn.LayerNorm(self.frame_num*(self.layer_dim_list[i_layer]+2)))
            self.output_proj_ped.append(Mlp_resid(self.frame_num*(self.layer_dim_list[i_layer]+2),self.frame_num*(self.layer_dim_list[i_layer]+2),self.text_dim))

            if i_layer>0:
                self.objf_resid_down.append(nn.Sequential(nn.Conv2d(self.text_dim,self.text_dim,3,2,1),
                                                           nn.Conv2d(self.text_dim,self.text_dim,1,1,0)))
            self.pos_avg_pool.append(nn.AdaptiveAvgPool3d([2,self.ptHlist[i_layer],self.ptHlist[i_layer]*3]))

            
            # self.absolute_pos_objf_embed.append(nn.Parameter(torch.zeros(1, self.ptHlist[i_layer]*self.ptHlist[i_layer]*3, text_dim)))
            # trunc_normal_(self.absolute_pos_objf_embed[-1], std=.02)

            self.head.append(Mlp(self.text_dim*self.qnum,self.text_dim*self.qnum,2,bias=False))

        self.sample_expression_num = sample_expression_num
        self.init_embeds = nn.Parameter(torch.zeros(((self.frame_num)*4,self.qnum,self.text_dim)))
        nn.init.normal_(self.init_embeds)

        self.absolute_lang_pos_embed = nn.Parameter(torch.zeros(1, self.text_len, self.text_dim))

        '''ESI+HMSI: per-layer modules that denoise an object-level caption against
        the trajectory visual feature and gate-fuse it into a holistic token,
        injected as a 4th K/V stream. Gated on cfg.ESI_ENABLED -> no params, no-op
        when disabled (vanilla FlexHook). Caption encoded by the frozen text encoder.'''
        self.esi_enabled = bool(getattr(cfg, 'ESI_ENABLED', False))
        self.esi_kv_len = int(getattr(cfg, 'ESI_KV_LEN', 1))
        if self.esi_enabled:
            self.hmsi_cap_norm = nn.ModuleList()
            self.hmsi_cap_proj = nn.ModuleList()
            self.hmsi_ca = nn.ModuleList()
            self.hmsi_gate = nn.ModuleList()
            self.hmsi_fuse_norm = nn.ModuleList()
            self.hmsi_out = nn.ModuleList()
            for i_layer in range(self.num_layers):
                self.hmsi_cap_norm.append(nn.LayerNorm(self.text_dim))
                self.hmsi_cap_proj.append(Mlp_resid(self.text_dim, self.text_dim, self.text_dim))
                self.hmsi_ca.append(CATransformerBlockTest(layer_id=i_layer, dim=self.text_dim, n_heads=8, norm_eps=None, drop_out=0.0))
                self.hmsi_gate.append(nn.Linear(self.text_dim * 2, self.text_dim))
                self.hmsi_fuse_norm.append(nn.LayerNorm(self.text_dim))
                self.hmsi_out.append(Mlp_resid(self.text_dim, self.text_dim, self.text_dim))

        '''L1 query-conditioned object representation (QCOND). Per layer, FiLM the
        otherwise text-blind obj_f by each expression's pooled text -> a per-(query,
        object) feature, appended block-diagonally. Gated on cfg.QCOND_ENABLED.
        gamma/beta zero-initialized => obj_f*(1+0)+0 == obj_f at step 0 (identity),
        so an untrained head is bit-identical to vanilla. No-op when disabled.'''
        self.qcond_enabled = bool(getattr(cfg, 'QCOND_ENABLED', False))
        self.qcond_residual = bool(getattr(cfg, 'QCOND_RESIDUAL', False))   # augment (keep shared obj_f) vs replace
        if self.qcond_enabled:
            self.qcond_text_norm = nn.ModuleList()
            self.qcond_gamma = nn.ModuleList()
            self.qcond_beta = nn.ModuleList()
            for i_layer in range(self.num_layers):
                self.qcond_text_norm.append(nn.LayerNorm(self.text_dim))
                g = nn.Linear(self.text_dim, self.text_dim); nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)
                bta = nn.Linear(self.text_dim, self.text_dim); nn.init.zeros_(bta.weight); nn.init.zeros_(bta.bias)
                self.qcond_gamma.append(g)
                self.qcond_beta.append(bta)

        if cfg.freeze_text:
            print('freeze text encoder !!!')
            for pn,p in  self.named_parameters():
                if pn.startswith('text_encoder'):
                    p.requires_grad_(False)

        if cfg.freeze_visual:
            print('freeze visual encoder !!!')
            for pn,p in  self.named_parameters():
                if pn.startswith('backbone'):
                    p.requires_grad_(False)

    def forward(self, x, pes,bbox_gt, expid,expma, cap_id=None, cap_mask=None):
        outs,l,text_mask,cap_feat = self.forward_features(x, expid,expma, cap_id, cap_mask)
        x = self.decode(outs,l,text_mask,pes,bbox_gt, cap_feat, cap_mask)
        return x

    def forward_features(self, inputs,expid,expma, cap_id=None, cap_mask=None):
        #print(exp)
        #pos = pes.flatten(0,1).permute(0,2,3,1) # bt,h,w,2
        #x = inputs[:,:,:3] 
        outputs=[]

        b,t,c,h,w = inputs.shape
        x = rearrange(inputs, 'B T C H W -> (B T) C H W')

        if hasattr(self.backbone,'forward_features'):
            outputs = self.backbone.forward_features(x)
        else:
            outputs = self.backbone(x)
        if isinstance(outputs,dict):
            outputs = [v for k,v in outputs.items()]

        # print([i.shape for i in outputs])
        
# torch.Size([2, 28224, 96])
# torch.Size([2, 7056, 192])
# torch.Size([2, 1764, 384])
# torch.Size([2, 441, 768])
# torch.Size([2, 441, 768])

        expid = expid.flatten(0,1)
        expma = expma.flatten(0,1)

        if hasattr(self.text_encoder,'encode_text_cpany'):
            encoded_text = self.text_encoder.encode_text_cpany(expid)
            expma = expma[:,:25]
        else:
            encoded_text = self.text_encoder(expid,expma).last_hidden_state

        encoded_text = rearrange(encoded_text, '(B N) L C -> B N L C',B=b,N=self.sample_expression_num,L=self.text_len)
        text_mask = rearrange(expma, '(B N) L -> B N L',B=b,N=self.sample_expression_num,L=self.text_len)

        #rec_q = encoded_text #b (tl) c

        '''ESI: encode the object-level caption (one per trajectory, no N axis)
        with the same frozen text encoder. cap_feat=(B, caption_len, C).'''
        cap_feat = None
        if self.esi_enabled and cap_id is not None:
            if hasattr(self.text_encoder,'encode_text_cpany'):
                cap_feat = self.text_encoder.encode_text_cpany(cap_id)
            else:
                cap_feat = self.text_encoder(cap_id, cap_mask).last_hidden_state

        return outputs,encoded_text,text_mask,cap_feat#,encoded_text

    def decode(self,outputs,text,text_mask,pos_raw,bbox_gt, cap_feat=None, cap_mask=None):

        n = self.sample_expression_num
        b,t,_,h,w = pos_raw.shape
        #print(pos_raw.shape)
        # assert 1==2
        q_out=0

        final_out = []

        prior = bbox_gt[:,:self.frame_num]

        scale_factor = 1 / math.sqrt(self.init_embeds.shape[-1])
        rec_q = (torch.einsum('bx,xqc->bqc',prior.flatten(1),self.init_embeds)*scale_factor).repeat(1,self.sample_expression_num,1)#.flatten(0,1) #BNC
        text_pos = self.absolute_lang_pos_embed.unsqueeze(1).repeat(b,self.sample_expression_num,1,1)#.flatten(1,2)#1LC->BNLC->B(NL)C
        # rec_q = (torch.einsum('bx,xqc->bqc',prior.flatten(1),self.init_embeds)*scale_factor).unsqueeze(1).repeat(1,self.sample_expression_num,1,1).flatten(0,1) #BN Q C
        # text_pos = self.absolute_lang_pos_embed.repeat(b*self.sample_expression_num,1,1)

        conditional_pos_embed = (self.conditional_pos_embed/(self.conditional_pos_embed.norm().detach())).repeat(b*n,1,1) #bn,10,c
        regular = 0
        for i,output in enumerate(outputs):
            
            cur_pos_raw = self.pos_avg_pool[i](pos_raw)#.detach()
            b,t,_,qh,qw = cur_pos_raw.shape

            speed = cur_pos_raw[:,1:]-cur_pos_raw[:,:-1]
            speed = torch.cat([speed,cur_pos_raw[:,-1:]],dim=1)
            cur_pos_raw = cur_pos_raw.flatten(0,1)
            
            speed = speed.flatten(0,1)

            # print(cur_pos_raw.shape)
            obj_f = F.grid_sample(output,cur_pos_raw.permute(0,2,3,1),padding_mode='zeros',align_corners=False)

            obj_f = torch.cat([obj_f,speed],dim=1) 
            # obj_f = rearrange(obj_f.unsqueeze(1),'(B T) N C H W-> B N (H W) (T C)',B=b,T=t).repeat(1,n,1,1).flatten(0,1)

            obj_f = rearrange(obj_f,'(B T) C H W-> B (H W) (T C)',B=b,T=t)

            obj_f = self.output_obj_norm_ped[i](obj_f)
            obj_f = self.output_proj_ped[i](obj_f)

            conditional_attn_mask = text_mask.flatten(0,1).unsqueeze(1).unsqueeze(1).repeat(1,1,self.condition_num,1).bool()
            #for_condition = 
            conditional_pos_embed = self.conditional_pos_decoder[i](conditional_pos_embed, text.flatten(0,1), None, text_pos.flatten(0,1), conditional_attn_mask)
            conditional_pos = self.conditional_pos_projector[i](conditional_pos_embed).sigmoid()*2-1 #bn,10,2
            # print(conditional_pos.shape)
            conditional_pos = rearrange(conditional_pos.unsqueeze(0).repeat(t,1,1,1),'T (B N) D C->(B T) N D C',B=b,N=n) #b,n,10,2
            #print(conditional_pos.shape)
            # regular += abs(abs(conditional_pos)-0.5).mean()#+(1-abs(conditional_pos).mean())
            # regular += 1-abs(conditional_pos.unsqueeze(2)-conditional_pos.unsqueeze(3)).mean()
            regular += 0.01*(point_dispersion_loss(conditional_pos) + point_dispersion_loss(conditional_pos.permute(0,2,1,3)))/2
            # if self.training == False and i==2:
            #     print(conditional_pos[3]) #320,8,10,2
            conditional_f = F.grid_sample(output,conditional_pos,padding_mode='zeros',align_corners=True).permute(0,2,3,1) #(bt) C n 10
            #print(conditional_f.shape)
            conditional_f = self.conditional_f_norm[i](conditional_f) #B*N 10 C
            conditional_f = self.conditional_f_proj[i](conditional_f) #B*N 10 C
            #assert 1==2
            conditional_f = rearrange(conditional_f,'(B T) N D C -> B (N D) C T',B=b,T=t).mean(-1) #B N*10 C
            if i > 0:
                obj_f = obj_f+obj_f_for_resid

            if i < (len(outputs)-1):
                #obj_f_for_resid = obj_f.clone()
                obj_f_for_resid = rearrange(obj_f,'B (H W) C->B C H W',B=b,H=self.ptHlist[i],W=self.ptHlist[i]*3,C=self.text_dim)
                obj_f_for_resid = self.objf_resid_down[i](obj_f_for_resid)
                obj_f_for_resid = rearrange(obj_f_for_resid,'B C H W->B (H W) C')


            #pooltext = text.mean(1)
            '''ESI+HMSI: denoise the object-level caption against this trajectory's
            visual feature (obj_f), gate-fuse into a holistic token, append as a
            4th K/V stream visible to all N expressions. None -> vanilla 3-stream.'''
            caption_f = None
            if self.esi_enabled and cap_feat is not None:
                cap_l = self.hmsi_cap_proj[i](self.hmsi_cap_norm[i](cap_feat))            # (B, cap_len, C)
                cap_refined = self.hmsi_ca[i](cap_l, obj_f, None, None, None)             # (B, cap_len, C)
                m = cap_mask.unsqueeze(-1).float()                                       # (B, cap_len, 1)
                cap_vec = (cap_refined*m).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1e-6)  # (B,1,C)
                obj_vec = obj_f.mean(1,keepdim=True)                                     # (B,1,C)
                g = torch.sigmoid(self.hmsi_gate[i](torch.cat([obj_vec,cap_vec],dim=-1)))# (B,1,C)
                caption_f = self.hmsi_out[i](self.hmsi_fuse_norm[i](obj_vec + g*cap_vec))# (B,1,C)

            # --- L1 QCOND: FiLM-modulate obj_f by each expression's pooled text into a
            # per-(query,object) block-diagonal stream. Two modes:
            #   replace  (qcond_enabled, not residual): the conditioned block REPLACES the
            #            shared text-blind obj_f -> matcher loses the un-modulated path
            #            (empirically destroys color grounding).
            #   residual (qcond_residual): AUGMENT, not replace -> emit BOTH the shared
            #            obj_f (visible to all N, preserves fine color/attribute grounding)
            #            AND the conditioned block-diagonal stream (per-query association).
            # vanilla/ESI (not qcond): shared obj_f only, visible to all N.
            Lobj = obj_f.shape[1]
            qcond = self.qcond_enabled
            residual = qcond and self.qcond_residual
            if qcond:
                tm = text_mask.unsqueeze(-1).to(obj_f.dtype)                       # (B,N,text_len,1)
                t_n = (text*tm).sum(2)/tm.sum(2).clamp_min(1e-6)                   # (B,N,C) pooled text
                t_n = self.qcond_text_norm[i](t_n)
                gamma = self.qcond_gamma[i](t_n).unsqueeze(2)                      # (B,N,1,C)
                beta = self.qcond_beta[i](t_n).unsqueeze(2)                        # (B,N,1,C)
                cond_block = (obj_f.unsqueeze(1)*(1+gamma)+beta).flatten(1,2)      # (B,N*Lobj,C)

            if residual:
                obj_streams = [obj_f, cond_block]      # shared (visible-all) + per-query (block-diag)
            elif qcond:
                obj_streams = [cond_block]             # replace
            else:
                obj_streams = [obj_f]                  # vanilla / ESI shared

            streams = [text.flatten(1,2), conditional_f] + obj_streams
            pos_streams = [text_pos.flatten(1,2), torch.zeros_like(conditional_f)] + \
                          [torch.zeros_like(s) for s in obj_streams]
            if caption_f is not None:
                streams.append(caption_f); pos_streams.append(torch.zeros_like(caption_f))
            kv = torch.cat(streams, dim=1)
            kvpos = torch.cat(pos_streams, dim=1)

            cap_len = caption_f.shape[1] if caption_f is not None else 0
            obj_start = self.text_len*n + self.condition_num*n
            obj_total = sum(s.shape[1] for s in obj_streams)                       # Lobj / N*Lobj / (Lobj+N*Lobj)
            rec_atten_mask = torch.zeros((b,1,n, obj_start + obj_total + cap_len),device=rec_q.device)

            cond_off = obj_start + (Lobj if residual else 0)                       # start of block-diagonal block
            for j in range(self.sample_expression_num):
                rec_atten_mask[:,0,j,j*self.text_len:(j+1)*self.text_len]=text_mask[:,j]
                rec_atten_mask[:,0,j,n*self.text_len+j*self.condition_num:n*self.text_len+(j+1)*self.condition_num]=1
                if qcond:
                    # block-diagonal: expression j sees ONLY its own conditioned obj block
                    rec_atten_mask[:,0,j, cond_off+j*Lobj:cond_off+(j+1)*Lobj]=1

            if residual or not qcond:
                # shared object features visible to all N (vanilla / ESI / residual's shared path)
                rec_atten_mask[:,:,:, obj_start:obj_start+Lobj]=1
            if cap_len:
                # caption is object-level -> visible to all N expressions
                rec_atten_mask[:,:,:, obj_start+obj_total:]=1

            rec_atten_mask = rec_atten_mask.bool()
            rec_q = self.output_ca_qkv[i](rec_q,kv,
                                        None,kvpos,
                                        rec_atten_mask) #b*8,2,c

            
            # q_score = rearrange(rec_q,'(B N) Q C->B N (Q C)',B=b,N=n)
            q_score = rec_q
            score = self.head[i](q_score)
            

                #score = self.head[0](score)
            
            final_out.append(score)

        final_out = torch.stack(final_out,dim=1)

        return final_out,regular
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'backbone.absolute_pos_embed','absolute_lang_pos_embed','text_encoder.position_embeddings'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'backbone.rope_freqs', 'backbone.relative_position_bias_table'}
