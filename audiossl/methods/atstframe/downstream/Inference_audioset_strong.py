
from audiossl.methods.atstframe.downstream.comparison_models.models.frame_atst import FrameAST_base
from torch import nn
import torch
from audiossl.methods.atstframe.downstream.utils_as_strong.model_as_strong import LinearHead
import torch
import torchaudio
from audiossl.transforms.common import MinMax
from torchvision import transforms
import sys



Labels = ["/g/122z_qxw","/m/01280g","/m/012f08","/m/012n7d","/m/012ndj","/m/012xff","/m/0130jx","/m/014yck","/m/014zdl","/m/0150b9","/m/015jpf","/m/015lz1","/m/015p6","/m/0174k2","/m/018p4k","/m/018w8","/m/0195fx","/m/0199g","/m/019jd","/m/01b82r","/m/01b9nn","/m/01b_21","/m/01bjv","/m/01c194","/m/01d380","/m/01d3sd","/m/01g50p","/m/01g90h","/m/01h3n","/m/01h82_","/m/01h8n0","/m/01hnzm","/m/01hsr_","/m/01j3sz","/m/01j423","/m/01j4z9","/m/01jg02","/m/01jnbd","/m/01jt3m","/m/01lsmm","/m/01lynh","/m/01m2v","/m/01m4t","/m/01rd7k","/m/01s0vc","/m/01sb50","/m/01swy6","/m/01v_m0","/m/01w250","/m/01x3z","/m/01xq0k1","/m/01y3hg","/m/01yg9g","/m/01yrx","/m/01z47d","/m/01z5f","/m/02021","/m/020bb7","/m/0239kh","/m/023pjk","/m/023vsd","/m/0242l","/m/024dl","/m/025_jnm","/m/025rv6n","/m/025wky1","/m/0261r1","/m/026fgl","/m/027m70_","/m/0284vy3","/m/028ght","/m/02_41","/m/02_nn","/m/02bk07","/m/02bm9n","/m/02c8p","/m/02dgv","/m/02f9f_","/m/02fs_r","/m/02fxyj","/m/02g901","/m/02jz0l","/m/02l6bg","/m/02mfyn","/m/02mk9","/m/02p01q","/m/02p3nc","/m/02pjr4","/m/02qldy","/m/02rhddq","/m/02rlv9","/m/02rr_","/m/02rtxlg","/m/02x984l","/m/02y_763","/m/02yds9","/m/02z32qm","/m/02zsn","/m/030rvx","/m/0316dw","/m/032n05","/m/032s66","/m/034srq","/m/0395lw","/m/039jq","/m/03cczk","/m/03cl9h","/m/03dnzn","/m/03fwl","/m/03j1ly","/m/03k3r","/m/03kmc9","/m/03l9g","/m/03m9d0z","/m/03p19w","/m/03q5_w","/m/03qc9zr","/m/03qtwd","/m/03v3yw","/m/03vt0","/m/03w41f","/m/03wvsk","/m/03wwcy","/m/04229","/m/0463cq4","/m/046dlr","/m/04_sv","/m/04brg2","/m/04cvmfc","/m/04fgwm","/m/04fq5q","/m/04gxbd","/m/04gy_2","/m/04k94","/m/04qvtq","/m/04rlf","/m/04rmv","/m/04s8yn","/m/04zjc","/m/04zmvq","/m/053hz1","/m/056ks2","/m/05_wcq","/m/05kq4","/m/05mxj0q","/m/05rj2","/m/05tny_","/m/05x_td","/m/05zc1","/m/05zppz","/m/0641k","/m/0642b4","/m/068hy","/m/068zj","/m/06_fw","/m/06_y0by","/m/06bxc","/m/06bz3","/m/06d_3","/m/06h7j","/m/06hck5","/m/06hps","/m/06mb1","/m/06q74","/m/06wzb","/m/06xkwv","/m/073cg4","/m/078jl","/m/0790c","/m/07bgp","/m/07bjf","/m/07c52","/m/07cx4","/m/07jdr","/m/07k1x","/m/07m2kt","/m/07mzm6","/m/07n_g","/m/07p6fty","/m/07p6mqd","/m/07p7b8y","/m/07p9k1k","/m/07pb8fc","/m/07pbtc8","/m/07pc8lb","/m/07pczhz","/m/07pdhp0","/m/07pdjhy","/m/07pggtn","/m/07phhsh","/m/07phxs1","/m/07pjjrj","/m/07pjwq1","/m/07pl1bw","/m/07plct2","/m/07plz5l","/m/07pn_8q","/m/07pp8cl","/m/07pp_mv","/m/07ppn3j","/m/07pqc89","/m/07pqn27","/m/07prgkl","/m/07pt_g0","/m/07ptfmf","/m/07ptzwd","/m/07pws3f","/m/07pxg6y","/m/07pyf11","/m/07pyy8b","/m/07pzfmf","/m/07q0h5t","/m/07q0yl5","/m/07q2z82","/m/07q4ntr","/m/07q5rw0","/m/07q6cd_","/m/07q7njn","/m/07q8f3b","/m/07qb_dv","/m/07qc9xj","/m/07qcpgn","/m/07qcx4z","/m/07qdb04","/m/07qf0zm","/m/07qfgpx","/m/07qfr4h","/m/07qh7jl","/m/07qjznl","/m/07qjznt","/m/07qlf79","/m/07qlwh6","/m/07qmpdm","/m/07qn4z3","/m/07qn5dc","/m/07qnq_y","/m/07qqyl4","/m/07qrkrw","/m/07qs1cx","/m/07qsvvw","/m/07qv4k0","/m/07qv_x_","/m/07qw_06","/m/07qwdck","/m/07qwf61","/m/07qwyj0","/m/07qyrcz","/m/07qz6j3","/m/07r04","/m/07r10fb","/m/07r4gkf","/m/07r4k75","/m/07r4wb8","/m/07r5c2p","/m/07r5v4s","/m/07r660_","/m/07r67yg","/m/07r81j2","/m/07r_25d","/m/07r_80w","/m/07r_k2n","/m/07rbp7_","/m/07rc7d9","/m/07rcgpl","/m/07rdhzs","/m/07rgkc5","/m/07rgt08","/m/07rjwbb","/m/07rjzl8","/m/07rn7sz","/m/07rpkh9","/m/07rqsjt","/m/07rrh0c","/m/07rrlb6","/m/07rv4dm","/m/07rv9rh","/m/07rwj3x","/m/07rwm0c","/m/07ryjzk","/m/07s02z0","/m/07s04w4","/m/07s0dtb","/m/07s12q4","/m/07s2xch","/m/07s34ls","/m/07s8j8t","/m/07sk0jz","/m/07sq110","/m/07sr1lc","/m/07st88b","/m/07st89h","/m/07svc2k","/m/07swgks","/m/07sx8x_","/m/07szfh9","/m/07yv9","/m/081rb","/m/0838f","/m/083vt","/m/08j51y","/m/0912c9","/m/0939n_","/m/093_4n","/m/096m7z","/m/09b5t","/m/09ct_","/m/09d5_","/m/09ddx","/m/09f96","/m/09hlz4","/m/09l8g","/m/09ld4","/m/09x0r","/m/09xqv","/m/0_1c","/m/0_ksk","/m/0b_fwt","/m/0bcdqg","/m/0brhx","/m/0bt9lr","/m/0btp2","/m/0bzvm2","/m/0c1dj","/m/0c1tlg","/m/0c2wf","/m/0c3f7m","/m/0cdnk","/m/0ch8v","/m/0chx_","/m/0cmf2","/m/0d31p","/m/0d4wf","/m/0dgbq","/m/0dgw9r","/m/0dl83","/m/0dl9sf8","/m/0dv3j","/m/0dv5r","/m/0dxrf","/m/0f8s22","/m/0fqfqc","/m/0fw86","/m/0fx9l","/m/0g12c5","/m/0g6b5","/m/0ghcn6","/m/0gvgw0","/m/0gy1t2s","/m/0h0rv","/m/0h2mp","/m/0h9mv","/m/0hdsk","/m/0hgq8df","/m/0hsrw","/m/0j2kx","/m/0j6m2","/m/0jb2l","/m/0jbk","/m/0k4j","/m/0k5j","/m/0k65p","/m/0l14jd","/m/0l156k","/m/0l15bq","/m/0l7xg","/m/0llzx","/m/0ltv","/m/0lyf6","/m/0md09","/m/0ngt1","/m/0ytgt","/m/0zmy2j9","/t/dd00001","/t/dd00002","/t/dd00003","/t/dd00004","/t/dd00005","/t/dd00006","/t/dd00013","/t/dd00018","/t/dd00038","/t/dd00048","/t/dd00061","/t/dd00065","/t/dd00066","/t/dd00067","/t/dd00077","/t/dd00088","/t/dd00091","/t/dd00092","/t/dd00099","/t/dd00109","/t/dd00112","/t/dd00121","/t/dd00125","/t/dd00126","/t/dd00127","/t/dd00128","/t/dd00130","/t/dd00134","/t/dd00135","/t/dd00136","/t/dd00138","/t/dd00139","/t/dd00141","/t/dd00142","/t/dd00143"]






class InferenceAudioSetStrong(nn.Module):
    def __init__(self,ckpt_path):
        super().__init__()
        self.encoder = FrameAST_base()
        self.head = LinearHead(768, 407, use_norm=False, affine=False)
        self._load_ckpt(ckpt_path)
        self.transform = self._transform()

    def _transform(self):
        melspec_t = torchaudio.transforms.MelSpectrogram(
            16000, f_min=60, f_max=7800, hop_length=160, win_length=1024, n_fft=1024, n_mels=64)
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power",top_db=80)
        normalize = MinMax(min=-79.6482,max=50.6842)
        return transforms.Compose([melspec_t,
                                to_db,
                                normalize])

    
    def _load_ckpt(self,ckpt_path):
        s = torch.load(ckpt_path,map_location="cpu")
        state_dict = s["state_dict"]
        replaced_state_dict = {}
        for key in state_dict.keys():
            replaced_state_dict[key.replace("encoder.encoder","encoder")] =  state_dict[key]

        self.load_state_dict(replaced_state_dict)

    def predict(self,wav):
        """
        ==================================================
        args:
        wav: torch.tensor in the shape of [1,N] or [B,1,N] 
        """"""
        return:
             retured prediction in the shape of [1,407,T] or [B,407,T]
        """
        if len(wav.shape)==2: 
            wav = wav.unsqueeze(1)
        else:
            assert len(wav.shape) == 3
    

        mel = self.transform(wav)
        chunk_len=1001 #10 secnods, consistent with the length of positional embedding
        output = []

        total_len = mel.shape[-1]
        num_chunks = total_len // chunk_len + 1
        for i in range(num_chunks):

            start = i*chunk_len
            end = (i+1) * chunk_len
            if end > total_len:
                end = total_len
            if end>start:
                mel_chunk=mel[:,:,:,start:end]
                len_chunk = torch.tensor([mel_chunk.shape[-1]]).expand(mel.shape[0]).to(wav.device)

                output_chunk = self.encoder.get_intermediate_layers(mel_chunk,len_chunk,n=1,scene=False)

                output.append(output_chunk)
        output=torch.cat(output,dim=1)
        output = self.head(output)
        return output


if __name__ == "__main__":
    import sys
    ckpt_path  = sys.argv[1]
    m = InferenceAudioSetStrong(ckpt_path)
    wav = torch.randn(1,160000)
    output = m.predict(wav)


