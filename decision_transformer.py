import torch
import torch.nn as nn
import torch.nn.functional as F



class InputLayer(nn.Module): 
    def __init__(self, emb_dim:int = 384, max_ep_len, state_dim, act_dim):
        super(InputLayer, self).__init__() 
        self.emb_dim = emb_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        
        self.embed_timestep = nn.Embedding(max_ep_len, emb_dim)
        self.embed_return = torch.nn.Linear(1, emb_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, emb_dim)
        self.embed_action = torch.nn.Linear(self.act_dim, emb_dim)

        self.embed_ln = nn.LayerNorm(emb_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        #================================================================================================
        stacked_inputs = 
        #================================================================================================
        z = self.embed_ln(stacked_inputs)

        return z


class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, emb_dim:int = 384, head:int = 3, dropout:float= 0.):
        """ 
        引数:
            emb_dim: 埋め込み後のベクトルの長さ 
            head: ヘッドの数
            dropout: ドロップアウト率
        """
        super(MultiHeadSelfAttention, self).__init__() 
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5 # D_hの二乗根。qk^Tを割るための係数
        #================================================================================================
        self.muskfilter = #マスクするためのフィルタ
        self.musk = #マスクの行列
        #================================================================================================

        # 入力をq,k,vに埋め込むための線形層。 [式(6)] 
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False) 
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        # 式(7)にはないが、実装ではドロップアウト層も用いる 
        self.attn_drop = nn.Dropout(dropout)

        # MHSAの結果を出力に埋め込むための線形層。[式(10)]
        ## 式(10)にはないが、実装ではドロップアウト層も用いる 
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout) 
        )


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            z: MHSAへの入力。形状は、(B, N, D)。
                B: バッチサイズ、N:トークンの数、D:ベクトルの長さ
        返り値:
            out: MHSAの出力。形状は、(B, N, D)。[式(10)]
                B:バッチサイズ、N:トークンの数、D:埋め込みベクトルの長さ
        """

        batch_size, num_patch, _ = z.size()

        # 埋め込み [式(6)]
        ## (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q,k,vをヘッドに分ける [式(10)]
        ## まずベクトルをヘッドの個数(h)に分ける
        ## (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        ## Self-Attentionができるように、
        ## (バッチサイズ、ヘッド、トークン数、パッチのベクトル)の形に変更する 
        ## (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # 内積 [式(7)]
        ## (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)
        ## (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N) 
        dots = (q @ k_T) / self.sqrt_dh
        #================================================================================================
        ##マスクの行列計算
        #dots = dot(dots muskfilter) + musk
        #================================================================================================
        ## 列方向にソフトマックス関数
        attn = F.softmax(dots, dim=-1)
        ## ドロップアウト
        attn = self.attn_drop(attn)
        # 加重和 [式(8)]
        ## (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h) 
        out = attn @ v
        ## (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        ## (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層 [式(10)]
        ## (B, N, D) -> (B, N, D) 
        out = self.w_o(out)

        return out



class EncoderBlock(nn.Module): 
    def __init__(self, emb_dim:int = 384, head:int=8, hidden_dim:int = 384*4, dropout:float = 0.):
        super(EncoderBlock, self).__init__()

        self.ln1 = nn.LayerNorm(emb_dim)

        self.msa = MultiHeadSelfAttention(
            emb_dim = emb_dim, 
            head = head,
            dropout = dropout,
        ).to(device)

        self.ln2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential( 
            nn.Linear(emb_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, emb_dim), 
            nn.Dropout(dropout)
        ).to(device)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.msa(self.ln1(z)) + z
        out = self.mlp(self.ln2(out)) + out

        return out



class DecisionTransformer(nn.Module): 
    def __init__(self, emb_dim:int = 384, num_blocks:int = 48, head:int = 8, hidden_dim:int = 384*4, dropout:float = 0.01):
        super(DecisionTransformer, self).__init__()

        self.input_layer = InputLayer(
            emb_dim
        ).to(device)

        self.encoder = nn.Sequential(*[
            EncoderBlock(
                emb_dim = emb_dim,
                head = head,
                hidden_dim = hidden_dim,
                dropout = dropout
            ).to(device)
            for _ in range(num_blocks)
        ])

        self.mlp_head_class = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 3)
        ).to(device)

        self.mlp_head_num = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1)
        ).to(device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.encoder(out)
        pred_class = self.mlp_head(out)
        pred_num = self.mlp_head(out)

        return pred



device = torch.device('cuda:0')

batch_size, channel, height, width= 1, 1000, 32, 32
x = torch.randn(batch_size, channel, height, width).to(device)

decisionT = DecisionTransformer().to(device)
pred = decisionT(x)
